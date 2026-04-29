import os
import random
import time
import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.model import Earlyfusion, Latefusion, MLPfusion


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
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
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
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        pos_count = mask.sum(1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


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


class RegressorHead(nn.Module):
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


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feat):
        return self.net(feat)


class JigsawHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feat):
        return self.net(feat)


def _extract_modality_features(model, text, audio, vision):
    z_text = torch.zeros_like(text)
    z_audio = torch.zeros_like(audio)
    z_vision = torch.zeros_like(vision)

    _, text_feat_raw = model([text, z_audio, z_vision])
    _, audio_feat_raw = model([z_text, audio, z_vision])
    _, vision_feat_raw = model([z_text, z_audio, vision])

    return {
        "text": _flatten_feature(text_feat_raw),
        "audio": _flatten_feature(audio_feat_raw),
        "vision": _flatten_feature(vision_feat_raw),
    }


def _sentiment_to_class(labels):
    y = labels.view(-1)
    cls = torch.full((y.shape[0],), 1, dtype=torch.long, device=y.device)
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

    if enc_out_dim % hyp_params.jigsaw_num_splits != 0:
        raise ValueError(
            "Encoder output dim must be divisible by --jigsaw_num_splits. "
            f"Got {enc_out_dim} and {hyp_params.jigsaw_num_splits}."
        )

    aux = {}
    for modality in ("text", "audio", "vision"):
        aux[f"{modality}_encoder"] = Encoder(
            feature_dims[modality], embed_dim=embed_dim, hidden_dim=hidden_dim
        )
        aux[f"{modality}_proj"] = ProjectHead(
            embed_dim, hidden_dim, hyp_params.simmmdg_proj_dim
        )
        aux[f"{modality}_regressor"] = RegressorHead(
            enc_out_dim,
            hidden_dim=max(hidden_dim, enc_out_dim),
            output_dim=output_dim,
            dropout=hyp_params.out_dropout if hasattr(hyp_params, "out_dropout") else 0.3,
        )
        aux[f"{modality}_classifier"] = ClassifierHead(
            enc_out_dim,
            hidden_dim=max(hidden_dim, enc_out_dim),
            output_dim=3,
            dropout=hyp_params.out_dropout if hasattr(hyp_params, "out_dropout") else 0.3,
        )

    aux["fusion_regressor"] = RegressorHead(
        enc_out_dim * 3,
        hidden_dim=max(hidden_dim, enc_out_dim),
        output_dim=output_dim,
        dropout=hyp_params.out_dropout if hasattr(hyp_params, "out_dropout") else 0.3,
    )
    aux["fusion_classifier"] = ClassifierHead(
        enc_out_dim * 3,
        hidden_dim=max(hidden_dim, enc_out_dim),
        output_dim=3,
        dropout=hyp_params.out_dropout if hasattr(hyp_params, "out_dropout") else 0.3,
    )
    aux["jigsaw_classifier"] = JigsawHead(
        input_dim=enc_out_dim * 3,
        output_dim=hyp_params.jigsaw_samples,
        hidden_dim=hyp_params.jigsaw_hidden,
    )

    for src in ("text", "audio", "vision"):
        for dst in ("text", "audio", "vision"):
            if src == dst:
                continue
            aux[f"{src}_to_{dst}"] = EncoderTrans(enc_out_dim, enc_out_dim, hidden_dim)

    return nn.ModuleDict(aux)


def _forward_moosa(model, aux, text, audio, vision):
    raw_features = _extract_modality_features(model, text, audio, vision)
    embeddings = {
        modality: aux[f"{modality}_encoder"](raw_features[modality])
        for modality in ("text", "audio", "vision")
    }
    unimodal_preds = {
        modality: aux[f"{modality}_regressor"](embeddings[modality])
        for modality in embeddings
    }
    unimodal_logits = {
        modality: aux[f"{modality}_classifier"](embeddings[modality])
        for modality in embeddings
    }
    fused_feat = torch.cat(
        [embeddings["text"], embeddings["audio"], embeddings["vision"]], dim=1
    )
    fused_pred = aux["fusion_regressor"](fused_feat)
    fused_logits = aux["fusion_classifier"](fused_feat)
    return fused_pred, {
        "text": embeddings["text"],
        "audio": embeddings["audio"],
        "vision": embeddings["vision"],
        "fused": fused_feat,
    }, unimodal_preds, unimodal_logits, fused_logits


def _regression_loss_per_sample(pred, target):
    loss = F.l1_loss(pred, target, reduction="none")
    if loss.dim() > 1:
        loss = loss.mean(dim=1)
    return loss


def _normalized_l2_distance(pred, target):
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return torch.mean(torch.norm(pred - target, dim=1))


def _compute_masked_translation_loss(aux, embeddings, hyp_params):
    losses = []
    for src in ("text", "audio", "vision"):
        source = embeddings[src]
        mask = (torch.rand_like(source) < hyp_params.mask_ratio).to(source.dtype)
        masked_source = source * (1.0 - mask)
        for dst in ("text", "audio", "vision"):
            if src == dst:
                continue
            translated = aux[f"{src}_to_{dst}"](masked_source)
            losses.append(_normalized_l2_distance(translated, embeddings[dst]))

    if not losses:
        return torch.tensor(0.0, device=embeddings["text"].device)
    return torch.stack(losses).mean()


def _compute_contrastive_loss(aux, embeddings, labels, criterion):
    views = []
    for modality in ("text", "audio", "vision"):
        half_dim = embeddings[modality].shape[1] // 2
        views.append(aux[f"{modality}_proj"](embeddings[modality][:, :half_dim]))
    stacked = torch.stack(views, dim=1)
    return criterion(stacked, _sentiment_to_class(labels))


def _compute_explore_loss(embeddings):
    losses = []
    for modality in ("text", "audio", "vision"):
        value = embeddings[modality]
        half_dim = value.shape[1] // 2
        losses.append(
            -F.mse_loss(
                F.normalize(value[:, :half_dim], dim=1),
                F.normalize(value[:, half_dim:], dim=1),
            )
        )
    return torch.stack(losses).mean()


def _sample_jigsaw_permutations(num_parts, num_samples, seed):
    max_unique = math.factorial(num_parts)
    target = min(num_samples, max_unique)
    rng = random.Random(seed)
    permutations = set()
    base = list(range(num_parts))
    while len(permutations) < target:
        candidate = tuple(rng.sample(base, len(base)))
        permutations.add(candidate)
    return [list(p) for p in sorted(permutations)]


def _compute_jigsaw_loss(aux, embeddings, hyp_params):
    split_parts = []
    for modality in ("text", "audio", "vision"):
        split_parts.extend(
            torch.chunk(embeddings[modality], hyp_params.jigsaw_num_splits, dim=1)
        )

    combinations = []
    labels = []
    for label, perm in enumerate(hyp_params.jigsaw_permutations):
        concatenated = torch.cat([split_parts[idx] for idx in perm], dim=1)
        combinations.append(concatenated)
        labels.append(
            torch.full(
                (concatenated.shape[0],),
                label,
                dtype=torch.long,
                device=concatenated.device,
            )
        )

    combinations = torch.cat(combinations, dim=0)
    labels = torch.cat(labels, dim=0)
    logits = aux["jigsaw_classifier"](combinations)
    return F.cross_entropy(logits, labels)


def _compute_entropy_weighted_loss(
    reg_losses,
    branch_logits,
    cls_labels,
    hyp_params,
):
    branch_names = ("text", "audio", "vision", "fused")

    entropies = []
    entropy_min_terms = []
    cls_losses = []
    weighted_losses = []
    for name in branch_names:
        probs = F.softmax(branch_logits[name], dim=1)
        entropies.append(-Categorical(probs=probs).entropy().unsqueeze(1))
        entropy_min_terms.append(
            (-probs * torch.log(probs + 1e-5)).sum(dim=1).mean()
        )
        cls_losses.append(F.cross_entropy(branch_logits[name], cls_labels))
        weighted_losses.append(reg_losses[name].unsqueeze(1))

    entropy_scores = torch.cat(entropies, dim=1)
    branch_weights = F.softmax(
        entropy_scores / hyp_params.entropy_weight_temp,
        dim=1,
    )
    reg_matrix = torch.cat(weighted_losses, dim=1)
    weighted_reg_loss = torch.mean(torch.sum(branch_weights * reg_matrix, dim=1))
    cls_loss = torch.stack(cls_losses).mean()
    entropy_min_loss = (
        torch.stack(entropy_min_terms).mean() * hyp_params.entropy_min_weight
    )
    return weighted_reg_loss, cls_loss, entropy_min_loss


def _checkpoint_payload(model, aux, hyp_params):
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    return {
        "format": "moosa_v1",
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
        "jigsaw_num_splits": int(hyp_params.jigsaw_num_splits),
        "jigsaw_samples": int(hyp_params.jigsaw_samples),
        "jigsaw_hidden": int(hyp_params.jigsaw_hidden),
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
        jigsaw_num_splits=checkpoint.get("jigsaw_num_splits", 4),
        jigsaw_samples=checkpoint.get("jigsaw_samples", 128),
        jigsaw_hidden=checkpoint.get("jigsaw_hidden", 512),
    )

    model = _build_fusion_model(load_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    aux = _build_auxiliary_modules(
        load_params, checkpoint["feature_dims"], checkpoint["output_dim"]
    ).to(device)
    aux.load_state_dict(
        checkpoint["aux_state_dict"],
        strict=checkpoint.get("format") == "moosa_v1",
    )
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
        if isinstance(checkpoint, dict) and checkpoint.get("format") in {
            "moosa_v1",
            "simmmdg_v1",
        }:
            model, aux = _load_checkpoint_for_eval(checkpoint, device)
            return evaluate_test_only(model, aux, criterion, hyp_params, test_loader)
        raise ValueError(
            "Unsupported checkpoint format for MOOSA evaluation. "
            "Expected a legacy nn.Module or a MOOSA/SimMMDG checkpoint dict."
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
        sample_features = _extract_modality_features(
            model, sample_text, sample_audio, sample_vision
        )
    model.train()

    feature_dims = {
        modality: sample_features[modality].shape[-1]
        for modality in ("text", "audio", "vision")
    }
    total_parts = 3 * hyp_params.jigsaw_num_splits
    hyp_params.jigsaw_permutations = _sample_jigsaw_permutations(
        total_parts, hyp_params.jigsaw_samples, hyp_params.seed
    )
    hyp_params.jigsaw_samples = len(hyp_params.jigsaw_permutations)

    aux_modules = _build_auxiliary_modules(
        hyp_params, feature_dims, hyp_params.output_dim
    ).to(device)

    all_params = list(model.parameters()) + list(aux_modules.parameters())
    optimizer = getattr(optim, hyp_params.optim)(all_params, lr=hyp_params.lr)

    trainable_params = _count_trainable_params(model) + _count_trainable_params(aux_modules)
    optimizer_params = _count_optimizer_params(optimizer)
    print(
        f"Trainable params (fusion+MOOSA): {trainable_params}, Optimizer params: {optimizer_params}"
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
                preds, _, _, _, _ = _forward_moosa(model, aux, text, audio, vision)
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
            cls_labels = _sentiment_to_class(eval_attr)

            optimizer.zero_grad()
            fused_pred, embeddings, unimodal_preds, unimodal_logits, fused_logits = _forward_moosa(
                model, aux, text, audio, vision
            )

            reg_losses = {
                "text": _regression_loss_per_sample(unimodal_preds["text"], eval_attr),
                "audio": _regression_loss_per_sample(unimodal_preds["audio"], eval_attr),
                "vision": _regression_loss_per_sample(unimodal_preds["vision"], eval_attr),
                "fused": _regression_loss_per_sample(fused_pred, eval_attr),
            }
            branch_logits = {
                "text": unimodal_logits["text"],
                "audio": unimodal_logits["audio"],
                "vision": unimodal_logits["vision"],
                "fused": fused_logits,
            }

            weighted_reg_loss, cls_loss, entropy_min_loss = _compute_entropy_weighted_loss(
                reg_losses, branch_logits, cls_labels, hyp_params
            )
            translation_loss = _compute_masked_translation_loss(aux, embeddings, hyp_params)
            contrastive_loss = _compute_contrastive_loss(
                aux, embeddings, eval_attr, criterion_supcon
            )
            jigsaw_loss = _compute_jigsaw_loss(aux, embeddings, hyp_params)
            explore_loss = _compute_explore_loss(embeddings)

            combined_loss = (
                weighted_reg_loss
                + cls_loss
                + entropy_min_loss
                + hyp_params.jigsaw_ratio * jigsaw_loss
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
            proc_loss += reg_losses["fused"].mean().item() * batch_size
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

                preds, _, _, _, _ = _forward_moosa(model, aux, text, audio, vision)
                total_loss += criterion_reg(preds, eval_attr).item()
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_val_acc2_record = -float("inf")
    best_val_loss_at_best_acc = float("inf")
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

        duration = time.time() - start
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
            print(f"Best Epoch {best_epoch_idx}")
            print(
                "Best Val Metrics  | Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
                    best_val_loss_at_best_acc,
                    best_val_metrics["mae"],
                    best_val_metrics["acc2"],
                    best_val_metrics["f1"],
                )
            )
            print(
                "Best Test Metrics | Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
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
