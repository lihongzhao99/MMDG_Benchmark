import argparse
import itertools
import math
import os
import os.path as osp
import random
import sys
from torch.distributions import Categorical
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils import data_loader_1d
from utils.run_logging import (
    append_best_result,
    build_output_paths,
    build_run_name,
    build_task_name,
    infer_dg_mode,
    parse_hust_domain_args,
    write_run_header,
)
from models import MOOSA_Model as models

METHOD_NAME = "MOOSA"
DATASET_NAME = "HUST"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 8
log_interval = 10
l2_decay = 5e-4

BASE_DIR = PROJECT_DIR
DATA_DIR = osp.join(BASE_DIR, "data")
ROOT_VIB_PATH = osp.join(DATA_DIR, "Motor_Vib.mat")
ROOT_ASC_PATH = osp.join(DATA_DIR, "Motor_Aud.mat")
RESULT_LOG_DIR = osp.join(BASE_DIR, "outputs", "logs", "MOOSA")
CKPT_DIR = osp.join(BASE_DIR, "outputs", "models", "MOOSA")


class SupConLoss(nn.Module):
    """Supervised contrastive loss on features of shape [B, V, D]."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.dim() != 3:
            raise ValueError("features must be [B, V, D]")

        device = features.device
        batch_size, n_views, _ = features.shape
        features = F.normalize(features, dim=2)
        features = features.reshape(batch_size * n_views, -1)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        mask = mask.repeat(n_views, n_views)

        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        logits_mask = torch.ones_like(logits)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        valid_mask = (mask_sum > 0).float()
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-12)
        loss = -mean_log_prob_pos * valid_mask
        loss = loss.sum() / (valid_mask.sum() + 1e-12)
        return loss


def build_train_val_loaders_from_source(
    src_full_loader, batch_size, kwargs, val_per_class=200, seed=8
):
    features, labels = src_full_loader.dataset.tensors
    labels_np = labels.cpu().numpy()

    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []

    domains = np.unique(labels_np[:, 1])
    classes = np.unique(labels_np[:, 0])

    for d in domains:
        for c in classes:
            idx = np.where((labels_np[:, 0] == c) & (labels_np[:, 1] == d))[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            v = min(val_per_class, len(idx) // 2)
            val_indices.extend(idx[:v].tolist())
            train_indices.extend(idx[v:].tolist())

    train_fea = features[train_indices]
    train_lab = labels[train_indices]
    val_fea = features[val_indices]
    val_lab = labels[val_indices]

    train_set = torch.utils.data.TensorDataset(train_fea, train_lab)
    val_set = torch.utils.data.TensorDataset(val_fea, val_lab)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )

    print(
        f"Split source train/val done: train={len(train_set)}, "
        f"val={len(val_set)}, val_per_class={val_per_class}"
    )
    return train_loader, val_loader


def compute_entropy(probabilities):
    return (-probabilities * torch.log(probabilities + 1e-5)).sum(dim=1)


def compute_weighted_classification_loss(
    vib_pred,
    aud_pred,
    fusion_pred,
    cls_label,
    criterion_cls,
    entropy_weight_temp,
):
    vib_ce = criterion_cls(vib_pred, cls_label)
    aud_ce = criterion_cls(aud_pred, cls_label)
    fusion_ce = criterion_cls(fusion_pred, cls_label)

    vib_prob = F.softmax(vib_pred, dim=1)
    aud_prob = F.softmax(aud_pred, dim=1)
    fusion_prob = F.softmax(fusion_pred, dim=1)

    vib_entropy = Categorical(probs=vib_prob).entropy().unsqueeze(1)
    aud_entropy = Categorical(probs=aud_prob).entropy().unsqueeze(1)
    fusion_entropy = Categorical(probs=fusion_prob).entropy().unsqueeze(1)

    entropy_stack = -torch.cat((vib_entropy, aud_entropy, fusion_entropy), dim=1)
    entropy_weights = F.softmax(entropy_stack / entropy_weight_temp, dim=1)

    classification_loss = torch.mean(
        entropy_weights[:, 0] * vib_ce
        + entropy_weights[:, 1] * aud_ce
        + entropy_weights[:, 2] * fusion_ce
    )

    entropy_min_loss = (
        compute_entropy(vib_prob).mean()
        + compute_entropy(aud_prob).mean()
        + compute_entropy(fusion_prob).mean()
    ) / 3.0

    return classification_loss, entropy_min_loss


def build_jigsaw_batch(vib_parts, aud_parts, jigsaw_indices, device):
    parts = tuple(vib_parts) + tuple(aud_parts)
    all_combinations = list(itertools.permutations(parts, len(parts)))
    selected_combinations = [all_combinations[idx] for idx in jigsaw_indices]

    jigsaw_features = []
    jigsaw_labels = []
    batch_size = parts[0].size(0)

    for label, ordered_parts in enumerate(selected_combinations):
        concatenated = torch.cat(ordered_parts, dim=1)
        jigsaw_features.append(concatenated)
        jigsaw_labels.append(
            torch.full((batch_size,), label, dtype=torch.long, device=device)
        )

    jigsaw_features = torch.cat(jigsaw_features, dim=0)
    jigsaw_labels = torch.cat(jigsaw_labels, dim=0)
    return jigsaw_features, jigsaw_labels


def compute_masked_translation_loss(vib_embedding, aud_embedding, model, mask_ratio):
    mask_vib = torch.rand_like(vib_embedding) < mask_ratio
    masked_vib = vib_embedding.clone()
    masked_vib[mask_vib] = 0

    mask_aud = torch.rand_like(aud_embedding) < mask_ratio
    masked_aud = aud_embedding.clone()
    masked_aud[mask_aud] = 0

    vib_to_aud = model.vib_to_aud(masked_vib)
    aud_to_vib = model.aud_to_vib(masked_aud)

    vib_to_aud = F.normalize(vib_to_aud, dim=1)
    aud_to_vib = F.normalize(aud_to_vib, dim=1)
    vib_target = F.normalize(vib_embedding, dim=1)
    aud_target = F.normalize(aud_embedding, dim=1)

    v2a_loss = torch.norm(vib_to_aud - aud_target, dim=1).mean()
    a2v_loss = torch.norm(aud_to_vib - vib_target, dim=1).mean()
    return 0.5 * (v2a_loss + a2v_loss)


# def compute_explore_loss(vib_embedding, aud_embedding):
#     vib_dim = vib_embedding.size(1) // 2
#     aud_dim = aud_embedding.size(1) // 2
#     loss_e = -F.mse_loss(vib_embedding[:, :vib_dim], vib_embedding[:, vib_dim:])
#     loss_e = loss_e - F.mse_loss(aud_embedding[:, :aud_dim], aud_embedding[:, aud_dim:])
#     return loss_e / 2.0


def compute_explore_loss(vib_embedding, aud_embedding):
    vib_dim = vib_embedding.size(1) // 2
    aud_dim = aud_embedding.size(1) // 2
    
    v1 = F.normalize(vib_embedding[:, :vib_dim], p=2, dim=1)
    v2 = F.normalize(vib_embedding[:, vib_dim:], p=2, dim=1)
    
    a1 = F.normalize(aud_embedding[:, :aud_dim], p=2, dim=1)
    a2 = F.normalize(aud_embedding[:, aud_dim:], p=2, dim=1)
    
    loss_e = -F.mse_loss(v1, v2) - F.mse_loss(a1, a2)
    
    return loss_e / 2.0


def forward_moosa_step(
    model,
    vib_data,
    aud_data,
    cls_label,
    criterion_cls,
    criterion_contrast,
    jigsaw_indices,
    entropy_weight_temp,
    entropy_min_weight,
    jigsaw_ratio,
    alpha_trans,
    alpha_contrast,
    explore_loss_coeff,
    mask_ratio,
):
    outputs = model(vib_data, aud_data, return_features=True)

    classification_loss, entropy_min_loss = compute_weighted_classification_loss(
        outputs["vib_pred"],
        outputs["aud_pred"],
        outputs["fusion_pred"],
        cls_label,
        criterion_cls,
        entropy_weight_temp,
    )

    jigsaw_features, jigsaw_labels = build_jigsaw_batch(
        outputs["vib_parts"],
        outputs["aud_parts"],
        jigsaw_indices,
        vib_data.device,
    )
    jigsaw_pred = model.jigsaw_cls(jigsaw_features)
    jigsaw_loss = F.cross_entropy(jigsaw_pred, jigsaw_labels)

    translation_loss = compute_masked_translation_loss(
        outputs["vib_embedding"],
        outputs["aud_embedding"],
        model,
        mask_ratio,
    )

    contrast_input = torch.stack(
        [outputs["vib_proj"], outputs["aud_proj"]],
        dim=1,
    )
    contrast_loss = criterion_contrast(contrast_input, cls_label)

    explore_loss = compute_explore_loss(
        outputs["vib_embedding"], outputs["aud_embedding"]
    )

    total_loss = classification_loss
    total_loss = total_loss + entropy_min_weight * entropy_min_loss
    total_loss = total_loss + jigsaw_ratio * jigsaw_loss
    total_loss = total_loss + alpha_trans * translation_loss
    total_loss = total_loss + alpha_contrast * contrast_loss
    total_loss = total_loss + explore_loss_coeff * explore_loss

    loss_terms = {
        "cls": classification_loss,
        "ent": entropy_min_loss,
        "jigsaw": jigsaw_loss,
        "trans": translation_loss,
        "contrast": contrast_loss,
        "explore": explore_loss,
    }
    return outputs["fusion_pred"], total_loss, loss_terms


def evaluate_fusion(model, loader, cuda, source_eval=False):
    model.eval()
    correct = 0
    total = len(loader.dataset)

    with torch.no_grad():
        for batch_data, batch_label in loader:
            if cuda:
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()

            labels = batch_label[:, 0] if source_eval else batch_label
            vib_data = batch_data[:, :1024]
            aud_data = batch_data[:, 1024:]

            fusion_pred = model(vib_data, aud_data)
            correct += fusion_pred.argmax(dim=1).eq(labels).sum().item()

    return 100.0 * correct / total


def evaluate_target_domains(model, target_loaders, cuda):
    per_target_results = {}
    for domain, loader in target_loaders.items():
        acc = evaluate_fusion(model, loader, cuda, source_eval=False)
        per_target_results[int(domain)] = acc
        tqdm.write(f"Target[D{int(domain)}] -> Fusion: {acc:.2f}%")

    mean_target_acc = float(np.mean(list(per_target_results.values()))) if per_target_results else 0.0
    per_target_str = ", ".join([f"D{k}:{v:.2f}%" for k, v in per_target_results.items()])
    tqdm.write(f"Target Mean     -> Fusion: {mean_target_acc:.2f}% | PerTarget: {{{per_target_str}}}")
    return per_target_results, mean_target_acc


def train(
    model,
    src_loader,
    val_loader,
    target_loaders,
    iteration,
    lr,
    cuda,
    task_name,
    alpha_trans=0.1,
    alpha_contrast=3.0,
    temp=0.1,
    explore_loss_coeff=0.7,
    entropy_weight_temp=1.0,
    entropy_min_weight=0.001,
    jigsaw_ratio=1.0,
    jigsaw_num_splits=4,
    jigsaw_samples=128,
    mask_ratio=0.3,
):
    src_iter = iter(src_loader)
    best_val_acc = 0.0
    best_test_acc_at_val = 0.0
    best_per_target_at_val = {}

    permutation_count = math.factorial(jigsaw_num_splits * 2)
    if jigsaw_samples > permutation_count:
        raise ValueError(
            f"jigsaw_samples={jigsaw_samples} exceeds available permutations={permutation_count}"
        )
    jigsaw_indices = random.sample(range(permutation_count), jigsaw_samples)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
    criterion_cls = nn.CrossEntropyLoss(reduction="none")
    criterion_contrast = SupConLoss(temperature=temp)

    if cuda:
        criterion_cls = criterion_cls.cuda()
        criterion_contrast = criterion_contrast.cuda()

    pbar = tqdm(range(1, iteration + 1), desc=f"Training {task_name}", unit="iter")

    for i in pbar:
        model.train()
        learning_rate = lr / math.pow((1 + 10 * (i - 1) / iteration), 0.75)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        try:
            src_data, src_label = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            src_data, src_label = next(src_iter)

        if cuda:
            src_data = src_data.cuda()
            src_label = src_label.cuda()

        cls_label = src_label[:, 0]
        vib_data = src_data[:, :1024]
        aud_data = src_data[:, 1024:]

        optimizer.zero_grad()
        pred, loss, loss_terms = forward_moosa_step(
            model,
            vib_data,
            aud_data,
            cls_label,
            criterion_cls,
            criterion_contrast,
            jigsaw_indices,
            entropy_weight_temp,
            entropy_min_weight,
            jigsaw_ratio,
            alpha_trans,
            alpha_contrast,
            explore_loss_coeff,
            mask_ratio,
        )
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            train_acc = pred.argmax(dim=1).eq(cls_label).float().mean().item() * 100.0
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Cls": f"{loss_terms['cls'].item():.4f}",
                    "Ent": f"{loss_terms['ent'].item():.4f}",
                    "Jig": f"{loss_terms['jigsaw'].item():.4f}",
                    "Trans": f"{loss_terms['trans'].item():.4f}",
                    "Con": f"{loss_terms['contrast'].item():.4f}",
                    "Exp": f"{loss_terms['explore'].item():.4f}",
                    "Acc": f"{train_acc:.2f}%",
                    "BestVal": f"{best_val_acc:.2f}%",
                }
            )

        if i % (log_interval * 20) == 0:
            tqdm.write(f"\n[Iter {i}] Validation & Testing...")
            src_acc = evaluate_fusion(model, src_loader, cuda, source_eval=True)
            val_acc = evaluate_fusion(model, val_loader, cuda, source_eval=True)
            current_per_target, test_acc = evaluate_target_domains(model, target_loaders, cuda)

            tqdm.write(f"Source Accuracy -> Fusion: {src_acc:.2f}%")
            tqdm.write(f"Val Accuracy    -> Fusion: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc_at_val = test_acc
                best_per_target_at_val = current_per_target
                torch.save(model.state_dict(), osp.join(CKPT_DIR, f"best_model_{task_name}.pth"))
                tqdm.write(
                    f">>> New Best Val: {best_val_acc:.2f}% | "
                    f"MeanTest@BestVal: {best_test_acc_at_val:.2f}% (Model Saved)"
                )

    return best_val_acc, best_test_acc_at_val, best_per_target_at_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--class_num", type=int, default=6)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--l2_decay", type=float, default=5e-4)
    parser.add_argument("--val_per_class", type=int, default=200)
    parser.add_argument("--alpha_trans", type=float, default=0.1)
    parser.add_argument("--trans_hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--alpha_contrast", type=float, default=3.0)
    parser.add_argument("--explore_loss_coeff", type=float, default=0.7)
    parser.add_argument("--entropy_weight_temp", type=float, default=1.0)
    parser.add_argument("--entropy_min_weight", type=float, default=0.001)
    parser.add_argument("--jigsaw_ratio", type=float, default=1.0)
    parser.add_argument("--jigsaw_num_splits", type=int, default=4)
    parser.add_argument("--jigsaw_samples", type=int, default=128)
    parser.add_argument("--jigsaw_hidden_dim", type=int, default=256)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("-s", "--source_domain", nargs="+", default=["D1"],
                        help="Source domain(s), e.g. -s D1 D2 D3 or -s 1 2 3")
    parser.add_argument("-t", "--target_domain", nargs="+", default=["D2"],
                        help="Target domain(s), e.g. -t D4 or -t 4")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    iteration = args.iteration
    batch_size = args.batch_size
    lr = args.lr
    class_num = args.class_num
    seed = args.seed
    log_interval = args.log_interval
    l2_decay = args.l2_decay
    source_domains, target_domains = parse_hust_domain_args(args.source_domain, args.target_domain)
    sourcelist = np.array(source_domains)
    targetlist = np.array(target_domains)
    task_name = build_task_name(sourcelist, targetlist)
    dg_mode = infer_dg_mode(sourcelist)
    run_name = build_run_name(METHOD_NAME, DATASET_NAME, sourcelist, targetlist, seed, args.run_name)
    RESULT_LOG_DIR, CKPT_DIR, log_path = build_output_paths(
        __file__, DATASET_NAME, METHOD_NAME, dg_mode, run_name
    )
    write_run_header(log_path, args, METHOD_NAME, dg_mode, sourcelist, targetlist)

    print(f"\n>>> Running {task_name}: {sourcelist} -> {targetlist}")
    print(f"Log path: {log_path}")

    cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "random" in globals():
        random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}

    src_full_loader = data_loader_1d.load_training(
        sourcelist, False, batch_size, kwargs,
        root_vib=ROOT_VIB_PATH, root_asc=ROOT_ASC_PATH, class_num=class_num
    )
    src_loader, val_loader = build_train_val_loaders_from_source(
        src_full_loader,
        batch_size=batch_size,
        kwargs=kwargs,
        val_per_class=args.val_per_class,
        seed=seed,
    )
    target_loaders = {}
    for domain in targetlist.tolist():
        target_loaders[int(domain)] = data_loader_1d.load_testing(
            np.array([int(domain)]), False, batch_size, kwargs,
            root_vib=ROOT_VIB_PATH, root_asc=ROOT_ASC_PATH, class_num=class_num
        )

    model = models.MOOSA(
        num_classes=class_num,
        embedding_dim=args.embedding_dim,
        cls_hidden_dim=args.hidden_dim,
        trans_hidden_dim=args.trans_hidden_dim,
        proj_hidden_dim=args.hidden_dim,
        proj_out_dim=args.out_dim,
        jigsaw_hidden_dim=args.jigsaw_hidden_dim,
        jigsaw_num_splits=args.jigsaw_num_splits,
        jigsaw_classes=args.jigsaw_samples,
    )
    if cuda:
        model.cuda()

    best_val_acc, test_at_best_val, best_per_target_at_val = train(
        model,
        src_loader,
        val_loader,
        target_loaders,
        iteration,
        lr,
        cuda,
        task_name,
        alpha_trans=args.alpha_trans,
        alpha_contrast=args.alpha_contrast,
        temp=args.temp,
        explore_loss_coeff=args.explore_loss_coeff,
        entropy_weight_temp=args.entropy_weight_temp,
        entropy_min_weight=args.entropy_min_weight,
        jigsaw_ratio=args.jigsaw_ratio,
        jigsaw_num_splits=args.jigsaw_num_splits,
        jigsaw_samples=args.jigsaw_samples,
        mask_ratio=args.mask_ratio,
    )

    best_ckpt_path = osp.join(CKPT_DIR, f"best_model_{task_name}.pth")
    if osp.exists(best_ckpt_path):
        state_dict = torch.load(best_ckpt_path, map_location="cuda" if cuda else "cpu")
        model.load_state_dict(state_dict)

    per_target_results, per_target_mean = evaluate_target_domains(model, target_loaders, cuda)
    if not best_per_target_at_val:
        best_per_target_at_val = per_target_results
        test_at_best_val = per_target_mean

    per_target_str = ", ".join([f"{k}:{v:.2f}%" for k, v in per_target_results.items()])
    best_per_target_str = ", ".join([f"{k}:{v:.2f}%" for k, v in best_per_target_at_val.items()])
    log_entry = (
        f"{task_name}: Source={sourcelist.tolist()}, Target={targetlist.tolist()}, "
        f"Best Val(Fusion)={best_val_acc:.2f}%, Test@BestVal(Fusion)={test_at_best_val:.2f}%, "
        f"BestPerTarget(Fusion)={{{best_per_target_str}}}, "
        f"PerTarget(Fusion)={{{per_target_str}}}, MeanTarget(Fusion)={per_target_mean:.2f}%\n"
    )
    print(log_entry.strip())
    append_best_result(log_path, "", best_val_acc, test_at_best_val, best_per_target_at_val, test_at_best_val)

    print(f"Finished {task_name}. Result saved to {log_path}")
