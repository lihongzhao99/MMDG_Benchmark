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


class ConMeanShiftLoss(nn.Module):
    def __init__(self, temperature=0.25):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_embeddings, positive_embeddings):
        device = anchor_embeddings.device
        batch_size = anchor_embeddings.shape[0]

        anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)

        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=1, keepdim=True)
        sim_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T)

        mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        pos_sim = pos_sim / self.temperature
        sim_matrix = sim_matrix / self.temperature

        logits_max = torch.max(sim_matrix, dim=1, keepdim=True)[0]
        sim_matrix = sim_matrix - logits_max.detach()
        pos_sim = pos_sim - logits_max.detach()

        exp_sim = torch.exp(sim_matrix).masked_fill(mask, 0)
        exp_pos = torch.exp(pos_sim)

        log_prob = pos_sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + exp_pos + 1e-10)
        loss = -log_prob.mean()
        return loss


class NELProjectHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=1)


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


def _compute_mean_shift_embedding(embeddings, k=8):
    n = embeddings.shape[0]
    if n <= 1:
        return F.normalize(embeddings, dim=1)

    k = max(1, min(k, n - 1))
    embeddings_norm = F.normalize(embeddings, dim=1)
    sim = torch.matmul(embeddings_norm, embeddings_norm.t())

    _, knn_idx = torch.topk(sim, k=k + 1, dim=1, largest=True)
    knn_idx = knn_idx[:, 1:]

    neighbor_weight = 0.5 / k
    neighbors = embeddings[knn_idx]
    weighted = 0.5 * embeddings + neighbor_weight * neighbors.sum(dim=1)
    return F.normalize(weighted, dim=1)


def _compute_nee_loss(embeddings):
    n = embeddings.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=embeddings.device)

    c_auto = torch.matmul(embeddings.t(), embeddings) / n
    try:
        eigenvalues = torch.linalg.eigvalsh(c_auto)
        eigenvalues = eigenvalues[eigenvalues > 1e-6]
        if eigenvalues.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)
        return (eigenvalues * torch.log(eigenvalues + 1e-10)).sum()
    except Exception:
        return torch.tensor(0.0, device=embeddings.device)


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

    model.eval()
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_text = sample_batch["text"].to(device)
        sample_audio = sample_batch["audio"].to(device)
        sample_vision = sample_batch["vision"].to(device)
        _, sample_feat = model([sample_text, sample_audio, sample_vision])
        fused_dim = _flatten_feature(sample_feat).shape[-1]
    model.train()

    nel_modules = {
        "text_proj": NELProjectHead(
            input_dim=fused_dim,
            hidden_dim=max(fused_dim, 128),
            out_dim=hyp_params.nel_proj_dim,
        ).to(device),
        "audio_proj": NELProjectHead(
            input_dim=fused_dim,
            hidden_dim=max(fused_dim, 128),
            out_dim=hyp_params.nel_proj_dim,
        ).to(device),
        "vision_proj": NELProjectHead(
            input_dim=fused_dim,
            hidden_dim=max(fused_dim, 128),
            out_dim=hyp_params.nel_proj_dim,
        ).to(device),
    }

    all_params = list(model.parameters())
    for module in nel_modules.values():
        all_params += list(module.parameters())
    optimizer = getattr(optim, hyp_params.optim)(all_params, lr=hyp_params.lr)

    trainable_params = _count_trainable_params(model) + sum(
        _count_trainable_params(m) for m in nel_modules.values()
    )
    optimizer_params = _count_optimizer_params(optimizer)
    print(
        f"Trainable params (fusion+NEL): {trainable_params}, Optimizer params: {optimizer_params}"
    )
    if trainable_params != optimizer_params:
        print(
            "[Warning] Some trainable parameters may not be included in optimizer param groups."
        )

    criterion_reg = getattr(nn, hyp_params.criterion)()
    criterion_supcon = SupConLoss(temperature=hyp_params.temp_s)
    criterion_unsupcon = ConMeanShiftLoss(temperature=hyp_params.temp_u)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=hyp_params.when, factor=0.1, verbose=True
    )

    settings = {
        "model": model,
        "nel": nel_modules,
        "optimizer": optimizer,
        "criterion_reg": criterion_reg,
        "criterion_supcon": criterion_supcon,
        "criterion_unsupcon": criterion_unsupcon,
        "scheduler": scheduler,
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
    nel = settings["nel"]
    optimizer = settings["optimizer"]
    criterion_reg = settings["criterion_reg"]
    criterion_supcon = settings["criterion_supcon"]
    criterion_unsupcon = settings["criterion_unsupcon"]
    scheduler = settings["scheduler"]
    device = _get_compute_device(hyp_params)

    enable_nel = getattr(hyp_params, "enable_nel", False)

    def train(model, nel, optimizer):
        epoch_loss = 0.0
        epoch_size = 0
        model.train()
        for module in nel.values():
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
            preds, fused_feat = model([text, audio, vision])
            fused_feat = _flatten_feature(fused_feat)

            raw_loss = criterion_reg(preds, eval_attr)
            combined_loss = raw_loss

            if enable_nel:
                # Build modality-specific features (text/audio/vision) and two stochastic views.
                text_feat, audio_feat, vision_feat = _extract_modality_features(
                    model, text, audio, vision
                )

                text_v1 = nel["text_proj"](text_feat)
                audio_v1 = nel["audio_proj"](audio_feat)
                vision_v1 = nel["vision_proj"](vision_feat)

                text_v2 = nel["text_proj"](text_feat)
                audio_v2 = nel["audio_proj"](audio_feat)
                vision_v2 = nel["vision_proj"](vision_feat)

                view1 = torch.stack([text_v1, audio_v1, vision_v1], dim=1)  # [B, 3, D]
                view2 = torch.stack([text_v2, audio_v2, vision_v2], dim=1)  # [B, 3, D]
                emd_proj = torch.cat([view1, view2], dim=0)  # [2B, 3, D]

                # L_C on initial projected embeddings (Eq.9 style).
                cls_labels = _sentiment_to_class(eval_attr)
                labels_2v = torch.cat([cls_labels, cls_labels], dim=0)
                loss_supervised = criterion_supcon(emd_proj, labels_2v)

                # L_UNC on mean-shift embeddings (Eq.8 style).
                ms_projs = []
                n_mod = emd_proj.shape[1]
                for m in range(n_mod):
                    ms = _compute_mean_shift_embedding(emd_proj[:, m, :], k=hyp_params.k)
                    ms_projs.append(ms)
                ms_projs = torch.stack(ms_projs, dim=1)  # [2B, 3, D]

                bsz = text.shape[0]
                ms_view1 = ms_projs[:bsz].reshape(bsz, -1)
                ms_view2 = ms_projs[bsz:].reshape(bsz, -1)
                loss_unsupervised = criterion_unsupcon(ms_view1, ms_view2)

                # L_NEE on two-view fused representation.
                fused_view1 = F.normalize(fused_feat, dim=1)
                fused_view2 = F.normalize(
                    F.dropout(fused_feat, p=0.2, training=True), dim=1
                )
                fused_2v = torch.cat([fused_view1, fused_view2], dim=0)
                loss_nee = _compute_nee_loss(fused_2v)

                nel_loss = (
                    hyp_params.alpha * loss_unsupervised
                    + (1.0 - hyp_params.alpha) * loss_supervised
                    + hyp_params.beta * loss_nee
                )
                combined_loss = raw_loss + hyp_params.nel_loss_weight * nel_loss

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
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

    def evaluate(model, nel, test=False):
        model.eval()
        for module in nel.values():
            module.eval()
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
    best_state_dict = None

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_loss = train(model, nel, optimizer)
        val_loss, val_r, val_t = evaluate(model, nel, test=False)
        test_loss, test_r, test_t = evaluate(model, nel, test=True)

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

    test_loss, r, t = evaluate(model, nel, test=True)
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
