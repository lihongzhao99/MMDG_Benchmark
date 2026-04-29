import torch
from torch import nn
from utils.util import *
import torch.optim as optim
import time
import os
import numpy as np
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


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    device = _get_compute_device(hyp_params)

    if hyp_params.stage == "dg_test":
        if not hyp_params.pretrained_model:
            raise ValueError("--pretrained_model is required when --stage dg_test")
        model = torch.load(hyp_params.pretrained_model, map_location=device)
        model = model.to(device)
        criterion = getattr(nn, hyp_params.criterion)()
        return evaluate_test_only(model, criterion, hyp_params, test_loader)

    if hyp_params.backbone == "latefusion":
        model = Latefusion(
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
    elif hyp_params.backbone == "earlyfusion":
        model = Earlyfusion(
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
    elif hyp_params.backbone == "mlp":
        model = MLPfusion(
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
    else:
        raise ValueError(
            f"Unsupported backbone '{hyp_params.backbone}'. Use latefusion, earlyfusion, or mlp."
        )
    model = model.to(device)
    if getattr(hyp_params, "multi_gpu", False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    trainable_params = _count_trainable_params(model)
    optimizer_params = _count_optimizer_params(optimizer)
    print(
        f"Trainable params: {trainable_params}, Optimizer params: {optimizer_params}"
    )
    if trainable_params != optimizer_params:
        print(
            "[Warning] Some trainable parameters may not be included in optimizer param groups."
        )
    criterion = getattr(nn, hyp_params.criterion)()
    
    # === 修改 1：Scheduler 监控指标改为最大化 (mode="max") ===
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode="max", patience=hyp_params.when, factor=0.1, verbose=True
    # )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=hyp_params.when, factor=0.1
    )
    
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
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

def _flatten_feature(feat):
    if feat.dim() == 1:
        return feat.unsqueeze(-1)
    if feat.dim() == 2:
        return feat
    return feat.mean(dim=1)

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

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]
    device = _get_compute_device(hyp_params)

    def train(model, optimizer, criterion):
        epoch_loss = 0
        epoch_size = 0
        model.train()
        # num_batches = hyp_params.n_train // hyp_params.batch_size
        num_batches = hyp_params.n_train
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, batch in enumerate(train_loader):
            text, audio, vision, batch_Y = (
                batch["text"],
                batch["audio"],
                batch["vision"],
                batch["label"],
            )
            eval_attr = batch_Y.unsqueeze(-1)
            model.zero_grad()

            text = text.to(device)
            audio = audio.to(device)
            vision = vision.to(device)
            eval_attr = eval_attr.to(device)

            batch_size = text.size(0)

            preds, _ = model([text, audio, vision])
            raw_loss = criterion(preds, eval_attr)

            text_feat, audio_feat, vision_feat = _extract_modality_features(
                    model, text, audio, vision
                )
            
            video_norm = vision_feat.norm(p=2, dim=1).mean()
            audio_norm = audio_feat.norm(p=2, dim=1).mean()
            text_norm = text_feat.norm(p=2, dim=1).mean()
            feat_frac1 = video_norm / audio_norm
            feat_frac2 = video_norm / text_norm
            feat_frac3 = text_norm / audio_norm
            loss_RNA = ((feat_frac1 - 1) ** 2 + (feat_frac2 - 1) ** 2 + (feat_frac3 - 1) ** 2) / 3
            combined_loss = raw_loss + hyp_params.alpha_RNA * loss_RNA
            
            combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            epoch_size += batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / hyp_params.log_interval,
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0, 0
                start_time = time.time()

        return epoch_loss / max(epoch_size, 1)

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
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

                batch_size = text.size(0)

                preds, _ = model([text, audio, vision])
                total_loss += criterion(preds, eval_attr).item()

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    # === 修改 2：将比较基准改为 ACC2，初始值为负无穷大 ===
    best_val_acc2_record = -float('inf') 
    best_val_loss_at_best_acc = float('inf')
    best_model_saved = False
    
    best_epoch_idx = -1
    best_val_metrics = None
    best_test_metrics = None
    best_test_loss = 0.0
    best_state_dict = None
    # ===============================================

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss, val_r, val_t = evaluate(model, criterion, test=False)
        test_loss, test_r, test_t = evaluate(model, criterion, test=True)

        source_val_metrics = summarize_senti_metrics(val_r, val_t)
        target_test_metrics = summarize_senti_metrics(test_r, test_t)

        end = time.time()
        duration = end - start
        
        # === 修改 3：让 scheduler 监控 validation acc2 ===
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

        # === 修改 4：判断条件改为当前 val_acc2 是否大于历史最大 acc2 ===
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
            
            # 刷新当前最佳 epoch 的记录
            best_epoch_idx = epoch
            best_val_metrics = source_val_metrics
            best_test_metrics = target_test_metrics
            best_test_loss = test_loss
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            # =========================================================

        # 打印当前的 Best Epoch 及其指标
        if best_epoch_idx != -1:
            print(f"👉 [Current Best] Epoch {best_epoch_idx}")
            print(
                "   Best Val Metrics  | Loss {:.6f} | MAE {:.6f} | Acc2 {:.6f} | F1 {:.6f}".format(
                    best_val_loss_at_best_acc,  # 使用取得最佳 ACC2 时的 Loss
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