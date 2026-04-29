import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        pos_count = mask.sum(1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128, hidden_dim=256):
        super().__init__()
        out_dim = embed_dim * 2
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feat):
        return self.enc_net(feat)


class EncoderTrans(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feat):
        return self.enc_net(feat)


class ProjectHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feat):
        return F.normalize(self.head(feat), dim=1)


class FusionRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feat):
        return 3.0 * torch.tanh(self.net(feat))


def _extract_modality_features(model, text, audio, vision):
    # Modality-isolated features, consistent with existing MDJA/GMP code style.
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


def _sentiment_to_class(labels):
    # labels: [B] or [B,1], map to {0:neg,1:neu,2:pos}
    y = labels.view(-1)
    cls = torch.full_like(y, 1, dtype=torch.long)
    cls = torch.where(y > 0, torch.full_like(cls, 2), cls)
    cls = torch.where(y < 0, torch.full_like(cls, 0), cls)
    return cls


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


def _build_auxiliary_modules(hyp_params, feature_dims, output_dim):
    embed_dim = hyp_params.simmmdg_embed_dim
    hidden_dim = hyp_params.simmmdg_hidden_dim
    enc_out_dim = embed_dim * 2

    aux = {
        "text_encoder": Encoder(feature_dims["text"], embed_dim, hidden_dim),
        "audio_encoder": Encoder(feature_dims["audio"], embed_dim, hidden_dim),
        "vision_encoder": Encoder(feature_dims["vision"], embed_dim, hidden_dim),
        "text_proj": ProjectHead(embed_dim, hidden_dim, hyp_params.simmmdg_proj_dim),
        "audio_proj": ProjectHead(embed_dim, hidden_dim, hyp_params.simmmdg_proj_dim),
        "vision_proj": ProjectHead(embed_dim, hidden_dim, hyp_params.simmmdg_proj_dim),
        "fusion_regressor": FusionRegressor(
            input_dim=enc_out_dim * 3,
            hidden_dim=max(hidden_dim, enc_out_dim),
            output_dim=output_dim,
            dropout=hyp_params.out_dropout if hasattr(hyp_params, "out_dropout") else 0.3,
        ),
    }

    for src in ("text", "audio", "vision"):
        for dst in ("text", "audio", "vision"):
            if src == dst:
                continue
            aux[f"{src}_to_{dst}"] = EncoderTrans(enc_out_dim, enc_out_dim, hidden_dim)

    return nn.ModuleDict(aux)


def _forward_simmmdg(model, aux, text, audio, vision):
    text_feat, audio_feat, vision_feat = _extract_modality_features(
        model, text, audio, vision
    )

    text_emd = aux["text_encoder"](text_feat)
    audio_emd = aux["audio_encoder"](audio_feat)
    vision_emd = aux["vision_encoder"](vision_feat)

    fused_feat = torch.cat([text_emd, audio_emd, vision_emd], dim=1)
    preds = aux["fusion_regressor"](fused_feat)

    return preds, {
        "text": text_emd,
        "audio": audio_emd,
        "vision": vision_emd,
        "fused": fused_feat,
    }


def _normalized_l2_distance(pred, target):
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return torch.mean(torch.norm(pred - target, dim=1))


def _compute_translation_loss(aux, embeddings):
    modality_names = ("text", "audio", "vision")
    losses = []
    for src in modality_names:
        for dst in modality_names:
            if src == dst:
                continue
            translated = aux[f"{src}_to_{dst}"](embeddings[src])
            losses.append(_normalized_l2_distance(translated, embeddings[dst]))
    if not losses:
        return torch.tensor(0.0, device=embeddings["text"].device)
    return torch.stack(losses).mean()


def _compute_contrastive_loss(aux, embeddings, labels, criterion):
    half_dim = embeddings["text"].shape[1] // 2
    text_proj = aux["text_proj"](embeddings["text"][:, :half_dim])
    audio_proj = aux["audio_proj"](embeddings["audio"][:, :half_dim])
    vision_proj = aux["vision_proj"](embeddings["vision"][:, :half_dim])
    stacked = torch.stack([text_proj, audio_proj, vision_proj], dim=1)
    cls_labels = _sentiment_to_class(labels)
    return criterion(stacked, cls_labels)


def _compute_explore_loss(embeddings):
    losses = []
    for value in embeddings.values():
        if value.ndim != 2:
            continue
        half_dim = value.shape[1] // 2
        # losses.append(-F.mse_loss(value[:, :half_dim], value[:, half_dim:]))
        losses.append(-F.mse_loss(F.normalize(value[:, :half_dim], dim=1), F.normalize(value[:, half_dim:], dim=1)))
    if not losses:
        return torch.tensor(0.0, device=next(iter(embeddings.values())).device)
    return torch.stack(losses).mean()


def _checkpoint_payload(model, aux, hyp_params):
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    return {
        "format": "simmmdg_v1",
        "backbone": hyp_params.backbone,
        "model_state_dict": model_to_save.state_dict(),
        "aux_state_dict": aux.state_dict(),
        "feature_dims": {
            "text": int(aux["text_encoder"].enc_net[0].in_features),
            "audio": int(aux["audio_encoder"].enc_net[0].in_features),
            "vision": int(aux["vision_encoder"].enc_net[0].in_features),
        },
        "orig_dim": list(hyp_params.orig_dim),
        "output_dim": int(hyp_params.output_dim),
        "proj_dim": int(hyp_params.proj_dim),
        "num_heads": int(hyp_params.num_heads),
        "layers": int(hyp_params.layers),
        "relu_dropout": float(hyp_params.relu_dropout),
        "embed_dropout": float(hyp_params.embed_dropout),
        "res_dropout": float(hyp_params.res_dropout),
        "out_dropout": float(hyp_params.out_dropout),
        "attn_dropout": float(hyp_params.attn_dropout),
        "simmmdg_embed_dim": int(hyp_params.simmmdg_embed_dim),
        "simmmdg_hidden_dim": int(hyp_params.simmmdg_hidden_dim),
        "simmmdg_proj_dim": int(hyp_params.simmmdg_proj_dim),
    }
    

def _load_checkpoint_for_eval(checkpoint, device):
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    load_params = AttrDict(
        backbone=checkpoint["backbone"],
        orig_dim=checkpoint["orig_dim"],
        output_dim=checkpoint["output_dim"],
        proj_dim=checkpoint["proj_dim"],
        num_heads=checkpoint["num_heads"],
        layers=checkpoint["layers"],
        relu_dropout=checkpoint["relu_dropout"],
        embed_dropout=checkpoint["embed_dropout"],
        res_dropout=checkpoint["res_dropout"],
        out_dropout=checkpoint["out_dropout"],
        attn_dropout=checkpoint["attn_dropout"],
        simmmdg_embed_dim=checkpoint["simmmdg_embed_dim"],
        simmmdg_hidden_dim=checkpoint["simmmdg_hidden_dim"],
        simmmdg_proj_dim=checkpoint["simmmdg_proj_dim"],
    )

    model = _build_fusion_model(load_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    aux = _build_auxiliary_modules(
        load_params, checkpoint["feature_dims"], checkpoint["output_dim"]
    ).to(
        device
    )
    aux.load_state_dict(checkpoint["aux_state_dict"])
    return model, aux


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    device = _get_compute_device(hyp_params)

    if hyp_params.stage == "dg_test":
        if not hyp_params.pretrained_model:
            raise ValueError("--pretrained_model is required when --stage dg_test")
        criterion = getattr(nn, hyp_params.criterion)()
        checkpoint = torch.load(hyp_params.pretrained_model, map_location=device)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint.to(device)
            return evaluate_test_only(model, None, criterion, hyp_params, test_loader)
        if isinstance(checkpoint, dict) and checkpoint.get("format") == "simmmdg_v1":
            model, aux = _load_checkpoint_for_eval(checkpoint, device)
            model = model.to(device)
            aux = aux.to(device)
            return evaluate_test_only(model, aux, criterion, hyp_params, test_loader)
        raise ValueError(
            "Unsupported checkpoint format for SimMMDG evaluation. "
            "Expected a legacy nn.Module or a SimMMDG checkpoint dict."
        )

    model = _build_fusion_model(hyp_params).to(device)
    if getattr(hyp_params, "multi_gpu", False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_text = sample_batch["text"].to(device)
        sample_audio = sample_batch["audio"].to(device)
        sample_vision = sample_batch["vision"].to(device)
        sample_text_feat, sample_audio_feat, sample_vision_feat = _extract_modality_features(
            model, sample_text, sample_audio, sample_vision
        )
    model.train()

    feature_dims = {
        "text": sample_text_feat.shape[-1],
        "audio": sample_audio_feat.shape[-1],
        "vision": sample_vision_feat.shape[-1],
    }
    aux_modules = _build_auxiliary_modules(
        hyp_params, feature_dims, hyp_params.output_dim
    ).to(device)

    all_params = list(model.parameters())
    all_params += list(aux_modules.parameters())
    optimizer = getattr(optim, hyp_params.optim)(all_params, lr=hyp_params.lr)

    trainable_params = _count_trainable_params(model) + sum(
        _count_trainable_params(m) for m in aux_modules.values()
    )
    optimizer_params = _count_optimizer_params(optimizer)
    print(
        f"Trainable params (fusion+SimMMDG): {trainable_params}, Optimizer params: {optimizer_params}"
    )
    if trainable_params != optimizer_params:
        print(
            "[Warning] Some trainable parameters may not be included in optimizer param groups."
        )

    criterion_reg = getattr(nn, hyp_params.criterion)()
    criterion_supcon = SupConLoss(temperature=hyp_params.contrast_temp)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=hyp_params.when, factor=0.1
    )

    settings = {
        "model": model,
        "aux": aux_modules,
        "optimizer": optimizer,
        "criterion_reg": criterion_reg,
        "criterion_supcon": criterion_supcon,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def evaluate_test_only(model, aux, criterion, hyp_params, test_loader):
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

            if aux is None:
                preds, _ = model([text, audio, vision])
            else:
                preds, _ = _forward_simmmdg(model, aux, text, audio, vision)
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
    aux = settings["aux"]
    optimizer = settings["optimizer"]
    criterion_reg = settings["criterion_reg"]
    criterion_supcon = settings["criterion_supcon"]
    scheduler = settings["scheduler"]
    device = _get_compute_device(hyp_params)

    def train(model, aux, optimizer):
        epoch_loss = 0.0
        epoch_size = 0
        model.train()
        aux.train()

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
            preds, embeddings = _forward_simmmdg(model, aux, text, audio, vision)

            raw_loss = criterion_reg(preds, eval_attr)
            translation_loss = _compute_translation_loss(aux, embeddings)
            contrastive_loss = _compute_contrastive_loss(
                aux, embeddings, eval_attr, criterion_supcon
            )
            explore_loss = _compute_explore_loss(
                {
                    "text": embeddings["text"],
                    "audio": embeddings["audio"],
                    "vision": embeddings["vision"],
                }
            )
            combined_loss = (
                raw_loss
                + hyp_params.alpha_trans * translation_loss
                + hyp_params.alpha_contrast * contrastive_loss
                + hyp_params.explore_loss_coeff * explore_loss
            )

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(aux.parameters()), hyp_params.clip
            )
            optimizer.step()

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

    def evaluate(model, aux, test=False):
        model.eval()
        aux.eval()
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

                preds, _ = _forward_simmmdg(model, aux, text, audio, vision)
                total_loss += criterion_reg(preds, eval_attr).item()

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
    best_checkpoint = None

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_loss = train(model, aux, optimizer)
        val_loss, val_r, val_t = evaluate(model, aux, test=False)
        test_loss, test_r, test_t = evaluate(model, aux, test=True)

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
                torch.save(_checkpoint_payload(model, aux, hyp_params), hyp_params.name)

            best_val_acc2_record = source_val_metrics["acc2"]
            best_val_loss_at_best_acc = val_loss
            best_model_saved = True

            best_epoch_idx = epoch
            best_val_metrics = source_val_metrics
            best_test_metrics = target_test_metrics
            best_test_loss = test_loss
            best_checkpoint = _checkpoint_payload(model, aux, hyp_params)
            best_checkpoint["model_state_dict"] = {
                k: v.detach().cpu().clone()
                for k, v in best_checkpoint["model_state_dict"].items()
            }
            best_checkpoint["aux_state_dict"] = {
                k: v.detach().cpu().clone()
                for k, v in best_checkpoint["aux_state_dict"].items()
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

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint["model_state_dict"])
        aux.load_state_dict(best_checkpoint["aux_state_dict"])

    test_loss, r, t = evaluate(model, aux, test=True)
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
