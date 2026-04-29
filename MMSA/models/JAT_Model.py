import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.model import Latefusion, Earlyfusion, MLPfusion
from utils.dataloader import GLOBAL_DOMAIN_TO_ID


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


def _masked_mean_pool(x):
    # x: [B, T, D], padded steps are zeros
    mask = (x.abs().sum(dim=-1) > 0).float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


def _flatten_feature(feat):
    if feat.dim() == 1:
        return feat.unsqueeze(-1)
    if feat.dim() == 2:
        return feat
    return feat.mean(dim=1)


def _entropy_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class ProjectHead(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

    def forward(self, x):
        return self.proj(x)


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


class Discriminator(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
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

    # Run one forward pass to infer fused feature dim for the global discriminator.
    model.eval()
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_text = sample_batch["text"].to(device)
        sample_audio = sample_batch["audio"].to(device)
        sample_vision = sample_batch["vision"].to(device)
        _, sample_feat = model([sample_text, sample_audio, sample_vision])
        fused_dim = _flatten_feature(sample_feat).shape[-1]
    model.train()

    mdja_modules = {
        # Use model-dependent fused features as MDJA inputs so losses can update backbone.
        "text_proj": ProjectHead(fused_dim, hyp_params.project_out_dim).to(device),
        "audio_proj": ProjectHead(fused_dim, hyp_params.project_out_dim).to(device),
        "vision_proj": ProjectHead(fused_dim, hyp_params.project_out_dim).to(device),
        "text_reg": ModalityRegressor(fused_dim, hyp_params.project_out_dim).to(device),
        "audio_reg": ModalityRegressor(fused_dim, hyp_params.project_out_dim).to(device),
        "vision_reg": ModalityRegressor(fused_dim, hyp_params.project_out_dim).to(device),
        "modal_disc": Discriminator(
            hyp_params.project_out_dim,
            out_dim=3,
            hidden_dim=hyp_params.global_discriminator_hidden_dim,
        ).to(device),
        "domain_disc_local": Discriminator(
            hyp_params.project_out_dim,
            out_dim=max(len(hyp_params.source_datasets), 2),
            hidden_dim=hyp_params.global_discriminator_hidden_dim,
        ).to(device),
    }

    mdja_modules["domain_disc_global"] = Discriminator(
        fused_dim,
        out_dim=max(len(hyp_params.source_datasets), 2),
        hidden_dim=hyp_params.global_discriminator_hidden_dim,
    ).to(device)

    all_params = list(model.parameters())
    for module in mdja_modules.values():
        all_params += list(module.parameters())

    optimizer = getattr(optim, hyp_params.optim)(all_params, lr=hyp_params.lr)

    trainable_params = _count_trainable_params(model) + sum(
        _count_trainable_params(m) for m in mdja_modules.values()
    )
    optimizer_params = _count_optimizer_params(optimizer)
    print(
        f"Trainable params (fusion+MDJA): {trainable_params}, Optimizer params: {optimizer_params}"
    )
    if trainable_params != optimizer_params:
        print(
            "[Warning] Some trainable parameters may not be included in optimizer param groups."
        )

    criterion_reg = getattr(nn, hyp_params.criterion)()
    criterion_ce = nn.CrossEntropyLoss()
    
    # ---------------------------------------------------------
    # 修改 1：Scheduler 监控指标改为最大化 (mode="max")，因为追踪的是 acc2
    # ---------------------------------------------------------
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=hyp_params.when, factor=0.1, verbose=True
    )

    source_to_local = {name: idx for idx, name in enumerate(hyp_params.source_datasets)}
    domain_id_lut = torch.full((len(GLOBAL_DOMAIN_TO_ID),), -1, dtype=torch.long)
    for name, local_id in source_to_local.items():
        global_id = GLOBAL_DOMAIN_TO_ID[name]
        domain_id_lut[global_id] = local_id

    settings = {
        "model": model,
        "mdja": mdja_modules,
        "optimizer": optimizer,
        "criterion_reg": criterion_reg,
        "criterion_ce": criterion_ce,
        "scheduler": scheduler,
        "domain_id_lut": domain_id_lut.to(device),
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


def _compute_mdja_losses(
    model,
    mdja,
    criterion_reg,
    criterion_ce,
    text,
    audio,
    vision,
    eval_attr,
    domain_labels,
    domain_id_lut,
    hyp_params,
):
    # Full multimodal forward.
    preds, fused_feat = model([text, audio, vision])
    fused_loss = criterion_reg(preds, eval_attr)

    # Modality-isolated forwards: ensures MDJA auxiliary terms backprop into fusion model.
    z_text = torch.zeros_like(text)
    z_audio = torch.zeros_like(audio)
    z_vision = torch.zeros_like(vision)

    _, text_feat_raw = model([text, z_audio, z_vision])
    _, audio_feat_raw = model([z_text, audio, z_vision])
    _, vision_feat_raw = model([z_text, z_audio, vision])

    text_feat = _flatten_feature(text_feat_raw)
    audio_feat = _flatten_feature(audio_feat_raw)
    vision_feat = _flatten_feature(vision_feat_raw)

    text_pred = mdja["text_reg"](text_feat)
    audio_pred = mdja["audio_reg"](audio_feat)
    vision_pred = mdja["vision_reg"](vision_feat)

    text_loss = criterion_reg(text_pred, eval_attr)
    audio_loss = criterion_reg(audio_pred, eval_attr)
    vision_loss = criterion_reg(vision_pred, eval_attr)

    text_proj = mdja["text_proj"](text_feat)
    audio_proj = mdja["audio_proj"](audio_feat)
    vision_proj = mdja["vision_proj"](vision_feat)

    modal_inputs = torch.cat(
        [
            grad_reverse(text_proj, hyp_params.alpha_rev),
            grad_reverse(audio_proj, hyp_params.alpha_rev),
            grad_reverse(vision_proj, hyp_params.alpha_rev),
        ],
        dim=0,
    )
    bsz = text.shape[0]
    modal_labels = torch.cat(
        [
            torch.zeros(bsz, dtype=torch.long, device=text.device),
            torch.ones(bsz, dtype=torch.long, device=text.device),
            torch.full((bsz,), 2, dtype=torch.long, device=text.device),
        ],
        dim=0,
    )
    modal_logits = mdja["modal_disc"](modal_inputs)
    modal_adv_loss = criterion_ce(modal_logits, modal_labels)

    cls_entropy_weights = {
        "text": 1.0 / 3.0,
        "audio": 1.0 / 3.0,
        "vision": 1.0 / 3.0,
    }
    domain_adv_loss_local = torch.tensor(0.0, device=text.device)
    domain_adv_loss_global = torch.tensor(0.0, device=text.device)

    if hyp_params.enable_domain_adv:
        local_domain_labels = domain_id_lut[domain_labels]
        if torch.any(local_domain_labels < 0):
            raise ValueError(
                "Found domain labels outside source domains; check source_datasets setup."
            )

        text_domain_logits = mdja["domain_disc_local"](
            grad_reverse(text_proj, hyp_params.alpha_rev2)
        )
        audio_domain_logits = mdja["domain_disc_local"](
            grad_reverse(audio_proj, hyp_params.alpha_rev2)
        )
        vision_domain_logits = mdja["domain_disc_local"](
            grad_reverse(vision_proj, hyp_params.alpha_rev2)
        )

        text_domain_loss = criterion_ce(text_domain_logits, local_domain_labels)
        audio_domain_loss = criterion_ce(audio_domain_logits, local_domain_labels)
        vision_domain_loss = criterion_ce(vision_domain_logits, local_domain_labels)

        domain_adv_loss_local = (
            text_domain_loss + audio_domain_loss + vision_domain_loss
        ) / 3.0

        text_ent = _entropy_from_logits(text_domain_logits)
        audio_ent = _entropy_from_logits(audio_domain_logits)
        vision_ent = _entropy_from_logits(vision_domain_logits)

        temp = max(float(hyp_params.entropy_temp), 1e-6)
        exp_t = torch.exp(text_ent / temp)
        exp_a = torch.exp(audio_ent / temp)
        exp_v = torch.exp(vision_ent / temp)
        exp_sum = exp_t + exp_a + exp_v
        cls_entropy_weights = {
            "text": exp_t / exp_sum,
            "audio": exp_a / exp_sum,
            "vision": exp_v / exp_sum,
        }

        fused_flat = _flatten_feature(fused_feat)
        global_domain_logits = mdja["domain_disc_global"](
            grad_reverse(fused_flat, hyp_params.alpha_rev2)
        )
        domain_adv_loss_global = criterion_ce(global_domain_logits, local_domain_labels)

    # MDJA auxiliary regression term (entropy-weighted across modalities).
    aux_reg_loss = (
        cls_entropy_weights["text"] * text_loss
        + cls_entropy_weights["audio"] * audio_loss
        + cls_entropy_weights["vision"] * vision_loss
    )

    total_loss = (
        fused_loss
        + hyp_params.domain_adv_loss_global * domain_adv_loss_global
        + hyp_params.domain_adv_loss_local * domain_adv_loss_local
        + hyp_params.modal_adv_loss * modal_adv_loss
        + hyp_params.cls_loss * aux_reg_loss
    )

    return preds, total_loss, fused_loss


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    mdja = settings["mdja"]
    optimizer = settings["optimizer"]
    criterion_reg = settings["criterion_reg"]
    criterion_ce = settings["criterion_ce"]
    scheduler = settings["scheduler"]
    domain_id_lut = settings["domain_id_lut"]
    device = _get_compute_device(hyp_params)

    def train_epoch():
        epoch_total_loss = 0.0
        epoch_fused_loss = 0.0
        epoch_size = 0

        model.train()
        for module in mdja.values():
            module.train()

        num_batches = hyp_params.n_train
        proc_loss, proc_size = 0.0, 0
        start_time = time.time()

        for i_batch, batch in enumerate(train_loader):
            text, audio, vision, batch_Y, domain_Y = (
                batch["text"],
                batch["audio"],
                batch["vision"],
                batch["label"],
                batch["domain_label"],
            )
            eval_attr = batch_Y.unsqueeze(-1)

            text = text.to(device)
            audio = audio.to(device)
            vision = vision.to(device)
            eval_attr = eval_attr.to(device)
            domain_Y = domain_Y.to(device)

            optimizer.zero_grad()

            if getattr(hyp_params, "enable_mdja", False):
                _, combined_loss, raw_loss = _compute_mdja_losses(
                    model,
                    mdja,
                    criterion_reg,
                    criterion_ce,
                    text,
                    audio,
                    vision,
                    eval_attr,
                    domain_Y,
                    domain_id_lut,
                    hyp_params,
                )
            else:
                preds, _ = model([text, audio, vision])
                raw_loss = criterion_reg(preds, eval_attr)
                combined_loss = raw_loss

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            batch_size = text.size(0)
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_total_loss += combined_loss.item() * batch_size
            epoch_fused_loss += raw_loss.item() * batch_size
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

        return (
            epoch_total_loss / max(epoch_size, 1),
            epoch_fused_loss / max(epoch_size, 1),
        )

    def evaluate(loader):
        model.eval()
        for module in mdja.values():
            module.eval()

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
                total_loss += criterion_reg(preds, eval_attr).item()

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / max(len(loader), 1)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    # ---------------------------------------------------------
    # 修改 2：改用 best_val_acc2_record 替代 best_val_loss_record
    # ---------------------------------------------------------
    best_val_acc2_record = -float("inf")
    best_val_loss_at_best_acc = float("inf")  # 用于记录最佳 ACC2 时对应的 loss 以便打印
    best_model_saved = False

    best_epoch_idx = -1
    best_val_metrics = None
    best_test_metrics = None
    best_test_loss = 0.0
    best_state_dict = None

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_total_loss, train_fused_loss = train_epoch()
        val_loss, val_r, val_t = evaluate(valid_loader)
        test_loss, test_r, test_t = evaluate(test_loader)

        source_val_metrics = summarize_senti_metrics(val_r, val_t)
        target_test_metrics = summarize_senti_metrics(test_r, test_t)

        end = time.time()
        duration = end - start
        
        # ---------------------------------------------------------
        # 修改 3：让 scheduler 接收 acc2，以便在 acc2 进入平台期时衰减学习率
        # ---------------------------------------------------------
        scheduler.step(source_val_metrics["acc2"])

        print("-" * 60)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Train Total Loss {:5.4f} | Train Fused Loss {:5.4f}".format(
                epoch, duration, train_total_loss, train_fused_loss
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

        # ---------------------------------------------------------
        # 修改 4：判定条件改为验证集上的 acc2 突破历史最高记录
        # ---------------------------------------------------------
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

    test_loss, r, t = evaluate(test_loader)
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