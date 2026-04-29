import copy
import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.model import Latefusion, Earlyfusion, MLPfusion


def _get_compute_device(hyp_params):
    if hasattr(hyp_params, "device"):
        return hyp_params.device
    return torch.device("cuda:0" if hyp_params.use_cuda else "cpu")


def _count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_optimizer_params(optimizer):
    total = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                total += p.numel()
    return total


def _flatten_feature(feat):
    if feat.dim() == 1:
        return feat.unsqueeze(-1)
    if feat.dim() == 2:
        return feat
    return feat.mean(dim=1)


def _extract_modality_features(model, text, audio, vision):
    z_text = torch.zeros_like(text)
    z_audio = torch.zeros_like(audio)
    z_vision = torch.zeros_like(vision)

    _, text_feat_raw = model([text, z_audio, z_vision])
    _, audio_feat_raw = model([z_text, audio, z_vision])
    _, vision_feat_raw = model([z_text, z_audio, vision])

    text_feat = _flatten_feature(text_feat_raw)
    audio_feat = _flatten_feature(audio_feat_raw)
    vision_feat = _flatten_feature(vision_feat_raw)
    return text_feat, audio_feat, vision_feat


def _confidence_from_reg(preds, targets):
    return torch.exp(-torch.abs(preds - targets).mean())


def _compute_ratio_dict(conf_dict, eps=1e-6):
    keys = list(conf_dict.keys())
    if len(keys) <= 1:
        return {k: torch.tensor(1.0, device=conf_dict[k].device) for k in keys}

    ratio_dict = {}
    for key in keys:
        others = [conf_dict[k] for k in keys if k != key]
        others_mean = torch.stack(others).mean()
        ratio_dict[key] = conf_dict[key] / (others_mean + eps)
    return ratio_dict


def _apply_modality_dropout(feat, ratio, base_prob):
    if ratio <= 1.0:
        return feat

    p_drop = base_prob + (1.0 - base_prob) * torch.tanh(ratio - 1.0)
    p_keep = torch.clamp(1.0 - p_drop, min=0.0, max=1.0)
    bsz = feat.size(0)
    mask = torch.bernoulli(p_keep * torch.ones(bsz, device=feat.device)).unsqueeze(1)
    return feat * mask


def _update_ema(student, teacher, beta):
    with torch.no_grad():
        student_params = dict(student.named_parameters())
        teacher_params = dict(teacher.named_parameters())
        for name in teacher_params:
            teacher_params[name].data.mul_(beta).add_(student_params[name].data, alpha=1.0 - beta)

        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(teacher.named_buffers())
        for name in teacher_buffers:
            teacher_buffers[name].data.copy_(student_buffers[name].data)


def _grad_dot_wrt_feature(loss_a, loss_b, feat):
    grad_a = torch.autograd.grad(
        loss_a,
        feat,
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )[0]
    grad_b = torch.autograd.grad(
        loss_b,
        feat,
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )[0]
    return (grad_a * grad_b).sum(dim=-1).mean()


class ModalityRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class FusionRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def _build_fusion_model(hyp_params):
    if hyp_params.backbone == "latefusion":
        return Latefusion(
            hyp_params.orig_dim,
            hyp_params.output_dim,
            hyp_params.proj_dim,
            hyp_params.num_heads,
            hyp_params.layers,
            hyp_params.relu_dropout,
            hyp_params.embed_dropout,
            hyp_params.res_dropout,
            hyp_params.out_dropout,
            hyp_params.attn_dropout,
        )
    if hyp_params.backbone == "earlyfusion":
        return Earlyfusion(
            hyp_params.orig_dim,
            hyp_params.output_dim,
            hyp_params.proj_dim,
            hyp_params.num_heads,
            hyp_params.layers,
            hyp_params.relu_dropout,
            hyp_params.embed_dropout,
            hyp_params.res_dropout,
            hyp_params.out_dropout,
            hyp_params.attn_dropout,
        )
    if hyp_params.backbone == "mlp":
        return MLPfusion(
            hyp_params.orig_dim,
            hyp_params.output_dim,
            hyp_params.proj_dim,
            hyp_params.num_heads,
            hyp_params.layers,
            hyp_params.relu_dropout,
            hyp_params.embed_dropout,
            hyp_params.res_dropout,
            hyp_params.out_dropout,
            hyp_params.attn_dropout,
        )
    raise ValueError(
        f"Unsupported backbone '{hyp_params.backbone}'. Use latefusion, earlyfusion, or mlp."
    )


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    device = _get_compute_device(hyp_params)

    if hyp_params.stage == "dg_test":
        if not hyp_params.pretrained_model:
            raise ValueError("--pretrained_model is required when --stage dg_test")
        model = torch.load(hyp_params.pretrained_model, map_location=device)
        model = model.to(device)
        criterion = getattr(nn, hyp_params.criterion)()
        return evaluate_test_only(model, criterion, hyp_params, test_loader)

    model = _build_fusion_model(hyp_params).to(device)
    if getattr(hyp_params, "multi_gpu", False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    enable_mbcd = getattr(hyp_params, "enable_mbcd", False)

    mbcd_modules = {}
    ema_model = None
    ema_mbcd_modules = {}

    all_params = list(model.parameters())
    if enable_mbcd:
        model.eval()
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_text = sample_batch["text"].to(device)
            sample_audio = sample_batch["audio"].to(device)
            sample_vision = sample_batch["vision"].to(device)
            _, sample_feat = model([sample_text, sample_audio, sample_vision])
            fused_dim = _flatten_feature(sample_feat).shape[-1]
        model.train()

        mbcd_modules = {
            "text_ln": nn.LayerNorm(fused_dim).to(device),
            "audio_ln": nn.LayerNorm(fused_dim).to(device),
            "vision_ln": nn.LayerNorm(fused_dim).to(device),
            "text_reg": ModalityRegressor(fused_dim, hyp_params.mbcd_hidden_dim).to(device),
            "audio_reg": ModalityRegressor(fused_dim, hyp_params.mbcd_hidden_dim).to(device),
            "vision_reg": ModalityRegressor(fused_dim, hyp_params.mbcd_hidden_dim).to(device),
            "fusion_reg": FusionRegressor(3 * fused_dim, hyp_params.mbcd_hidden_dim).to(device),
        }

        for module in mbcd_modules.values():
            all_params += list(module.parameters())

        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_model.eval()

        for name, module in mbcd_modules.items():
            ema_mod = copy.deepcopy(module).to(device)
            for p in ema_mod.parameters():
                p.requires_grad = False
            ema_mod.eval()
            ema_mbcd_modules[name] = ema_mod

    optimizer = getattr(optim, hyp_params.optim)(all_params, lr=hyp_params.lr)

    trainable_params = _count_trainable_params(model) + sum(
        _count_trainable_params(m) for m in mbcd_modules.values()
    )
    optimizer_params = _count_optimizer_params(optimizer)
    print(
        f"Trainable params (fusion+MBCD): {trainable_params}, Optimizer params: {optimizer_params}"
    )
    if trainable_params != optimizer_params:
        print(
            "[Warning] Some trainable parameters may not be included in optimizer param groups."
        )

    criterion = getattr(nn, hyp_params.criterion)()
    criterion_distill = nn.MSELoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=hyp_params.when, factor=0.1, verbose=True
    )

    settings = {
        "model": model,
        "mbcd": mbcd_modules,
        "ema_model": ema_model,
        "ema_mbcd": ema_mbcd_modules,
        "optimizer": optimizer,
        "criterion": criterion,
        "criterion_distill": criterion_distill,
        "scheduler": scheduler,
        "enable_mbcd": enable_mbcd,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def evaluate_test_only(model, criterion, hyp_params, test_loader):
    device = _get_compute_device(hyp_params)
    model.eval()
    total_loss = 0.0
    results = []
    truths = []

    with torch.no_grad():
        for batch in test_loader:
            text, audio, vision, batch_Y = (
                batch["text"],
                batch["audio"],
                batch["vision"],
                batch["label"],
            )
            eval_attr = batch_Y.unsqueeze(-1)

            text = text.to(device)
            audio = audio.to(device)
            vision = vision.to(device)
            eval_attr = eval_attr.to(device)

            preds, _ = model([text, audio, vision])
            total_loss += criterion(preds, eval_attr).item()
            results.append(preds)
            truths.append(eval_attr)

    avg_loss = total_loss / max(hyp_params.n_test, 1)
    results = torch.cat(results)
    truths = torch.cat(truths)
    metrics = summarize_senti_metrics(results, truths)
    print("=" * 60)
    print("Frozen Target Test Metrics")
    print(
        "Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
            avg_loss,
            metrics["mae"],
            metrics["acc2"],
            metrics["f1"],
        )
    )
    print("=" * 60)
    return avg_loss


def summarize_senti_metrics(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )
    if non_zeros.size == 0:
        non_zeros = np.arange(len(test_truth))

    mae = np.mean(np.absolute(test_preds - test_truth))

    f_score = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    acc2 = accuracy_score(binary_truth, binary_preds)

    return {
        "mae": float(mae),
        "f1": float(f_score),
        "acc2": float(acc2),
    }


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    mbcd = settings["mbcd"]
    ema_model = settings["ema_model"]
    ema_mbcd = settings["ema_mbcd"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    criterion_distill = settings["criterion_distill"]
    scheduler = settings["scheduler"]
    enable_mbcd = settings["enable_mbcd"]
    device = _get_compute_device(hyp_params)

    def train(model, optimizer, criterion):
        epoch_loss = 0.0
        epoch_size = 0
        model.train()
        for module in mbcd.values():
            module.train()

        num_batches = hyp_params.n_train
        proc_loss, proc_size = 0.0, 0
        start_time = time.time()

        for i_batch, batch in enumerate(train_loader):
            text, audio, vision, batch_Y = (
                batch["text"],
                batch["audio"],
                batch["vision"],
                batch["label"],
            )
            eval_attr = batch_Y.unsqueeze(-1)

            text = text.to(device)
            audio = audio.to(device)
            vision = vision.to(device)
            eval_attr = eval_attr.to(device)

            optimizer.zero_grad()
            preds, _ = model([text, audio, vision])
            raw_loss = criterion(preds, eval_attr)
            combined_loss = raw_loss

            if enable_mbcd:
                text_feat, audio_feat, vision_feat = _extract_modality_features(
                    model, text, audio, vision
                )

                if getattr(hyp_params, "enable_layernorm", False):
                    text_feat = mbcd["text_ln"](text_feat)
                    audio_feat = mbcd["audio_ln"](audio_feat)
                    vision_feat = mbcd["vision_ln"](vision_feat)

                text_pred = mbcd["text_reg"](text_feat)
                audio_pred = mbcd["audio_reg"](audio_feat)
                vision_pred = mbcd["vision_reg"](vision_feat)

                text_loss = criterion(text_pred, eval_attr)
                audio_loss = criterion(audio_pred, eval_attr)
                vision_loss = criterion(vision_pred, eval_attr)
                modal_loss = (text_loss + audio_loss + vision_loss) / 3.0

                conf_dict = {
                    "text": _confidence_from_reg(text_pred, eval_attr).detach(),
                    "audio": _confidence_from_reg(audio_pred, eval_attr).detach(),
                    "vision": _confidence_from_reg(vision_pred, eval_attr).detach(),
                }
                ratio_dict = _compute_ratio_dict(conf_dict)

                text_feat_drop = _apply_modality_dropout(
                    text_feat,
                    ratio_dict["text"],
                    float(hyp_params.modality_drop_base),
                )
                audio_feat_drop = _apply_modality_dropout(
                    audio_feat,
                    ratio_dict["audio"],
                    float(hyp_params.modality_drop_base),
                )
                vision_feat_drop = _apply_modality_dropout(
                    vision_feat,
                    ratio_dict["vision"],
                    float(hyp_params.modality_drop_base),
                )

                fused_aux_feat = torch.cat(
                    [text_feat_drop, audio_feat_drop, vision_feat_drop], dim=1
                )
                fused_aux_pred = mbcd["fusion_reg"](fused_aux_feat)
                fused_aux_loss = criterion(fused_aux_pred, eval_attr)

                gcc_term = torch.tensor(0.0, device=device)
                if getattr(hyp_params, "enable_grad_consistency", False):
                    dot_t = _grad_dot_wrt_feature(text_loss, fused_aux_loss, text_feat)
                    dot_a = _grad_dot_wrt_feature(audio_loss, fused_aux_loss, audio_feat)
                    dot_v = _grad_dot_wrt_feature(vision_loss, fused_aux_loss, vision_feat)
                    # Eq.(6)-style objective: L - alpha * <g_uni, g_mm>
                    gcc_term = -float(hyp_params.gcc_alpha) * (dot_t + dot_a + dot_v) / 3.0

                kd_mm_loss = torch.tensor(0.0, device=device)
                kd_um_loss = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    ema_preds, _ = ema_model([text, audio, vision])

                if float(hyp_params.kl_mm_coeff) > 0:
                    kd_mm_loss = criterion_distill(preds, ema_preds)
                if float(hyp_params.kl_um_coeff) > 0:
                    kd_um_loss = (
                        criterion_distill(text_pred, ema_preds)
                        + criterion_distill(audio_pred, ema_preds)
                        + criterion_distill(vision_pred, ema_preds)
                    ) / 3.0

                combined_loss = (
                    raw_loss
                    + float(hyp_params.mbcd_modal_loss_weight) * modal_loss
                    + float(hyp_params.mbcd_fused_loss_weight) * fused_aux_loss
                    + float(hyp_params.gcc_weight) * gcc_term
                    + float(hyp_params.kl_mm_coeff) * kd_mm_loss
                    + float(hyp_params.kl_um_coeff) * kd_um_loss
                )

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            if enable_mbcd:
                _update_ema(model, ema_model, float(hyp_params.ema_beta))
                for key in mbcd.keys():
                    _update_ema(mbcd[key], ema_mbcd[key], float(hyp_params.ema_beta))

            batch_size = text.size(0)
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            epoch_size += batch_size

            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / max(proc_size, 1)
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Fused Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / max(hyp_params.log_interval, 1),
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0.0, 0
                start_time = time.time()

        return epoch_loss / max(epoch_size, 1)

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for batch in loader:
                text, audio, vision, batch_Y = (
                    batch["text"],
                    batch["audio"],
                    batch["vision"],
                    batch["label"],
                )
                eval_attr = batch_Y.unsqueeze(-1)

                text = text.to(device)
                audio = audio.to(device)
                vision = vision.to(device)
                eval_attr = eval_attr.to(device)

                preds, _ = model([text, audio, vision])
                total_loss += criterion(preds, eval_attr).item()

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_val_acc2_record = -float("inf")
    best_val_loss_at_best_acc = float("inf")
    best_model_saved = False

    best_epoch_idx = -1
    best_val_metrics = None
    best_test_metrics = None
    best_test_loss = 0.0
    best_state_dict = None

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss, val_r, val_t = evaluate(model, criterion, test=False)
        test_loss, test_r, test_t = evaluate(model, criterion, test=True)

        source_val_metrics = summarize_senti_metrics(val_r, val_t)
        target_test_metrics = summarize_senti_metrics(test_r, test_t)

        end = time.time()
        duration = end - start

        scheduler.step(source_val_metrics["acc2"])

        print("-" * 60)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f}".format(
                epoch, duration, train_loss
            )
        )
        print(
            "Source Val Metrics  | Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
                val_loss,
                source_val_metrics["mae"],
                source_val_metrics["acc2"],
                source_val_metrics["f1"],
            )
        )
        print(
            "Target Test Metrics | Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
                test_loss,
                target_test_metrics["mae"],
                target_test_metrics["acc2"],
                target_test_metrics["f1"],
            )
        )
        print("-" * 60)

        if source_val_metrics["acc2"] > best_val_acc2_record:
            if hyp_params.name:
                save_dir = os.path.dirname(hyp_params.name)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                print(f"*** New Best Model Saved at {hyp_params.name}! ***")
                torch.save(model, hyp_params.name)

            best_val_acc2_record = source_val_metrics["acc2"]
            best_val_loss_at_best_acc = val_loss
            best_model_saved = True

            best_epoch_idx = epoch
            best_val_metrics = source_val_metrics
            best_test_metrics = target_test_metrics
            best_test_loss = test_loss
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if best_epoch_idx != -1:
            print(f"👉 [Current Best] Epoch {best_epoch_idx}")
            print(
                "   Best Val Metrics  | Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
                    best_val_loss_at_best_acc,
                    best_val_metrics["mae"],
                    best_val_metrics["acc2"],
                    best_val_metrics["f1"],
                )
            )
            print(
                "   Best Test Metrics | Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
                    best_test_loss,
                    best_test_metrics["mae"],
                    best_test_metrics["acc2"],
                    best_test_metrics["f1"],
                )
            )
            print("-" * 60)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, r, t = evaluate(model, criterion, test=True)
    final_metrics = summarize_senti_metrics(r, t)
    print("=" * 50)
    print("Final Frozen Target Test Metrics (from Best Model)")
    print(
        "Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
            test_loss,
            final_metrics["mae"],
            final_metrics["acc2"],
            final_metrics["f1"],
        )
    )
    print("=" * 50)
    return {
        "best_epoch": int(best_epoch_idx),
        "best_val": best_val_metrics,
        "best_test": best_test_metrics,
        "best_val_loss": float(best_val_loss_at_best_acc),
        "best_test_loss": float(best_test_loss),
        "final_test_loss": float(test_loss),
        "final_test_metrics": final_metrics,
    }
