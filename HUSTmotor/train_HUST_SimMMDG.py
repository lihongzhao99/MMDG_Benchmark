import argparse
import math
import os
import os.path as osp
import sys
import random
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
from models import SimMMDG_Model as models

METHOD_NAME = "SimMMDG"
DATASET_NAME = "HUST"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 8
log_interval = 10
l2_decay = 5e-4

BASE_DIR = PROJECT_DIR
DATA_DIR = osp.join(BASE_DIR, "data")
ROOT_VIB_PATH = osp.join(DATA_DIR, "Motor_Vib.mat")
ROOT_ASC_PATH = osp.join(DATA_DIR, "Motor_Aud.mat")
RESULT_LOG_DIR = osp.join(BASE_DIR, "outputs", "logs", "SimMMDG")
CKPT_DIR = osp.join(BASE_DIR, "outputs", "models", "SimMMDG")


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


def build_train_val_loaders_from_source(src_full_loader, batch_size, kwargs, val_per_class=200, seed=8):
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


def compute_translation_loss(src_embedding, tgt_embedding, translator):
    translated = translator(src_embedding)
    translated = translated / (torch.norm(translated, dim=1, keepdim=True) + 1e-12)
    target = tgt_embedding / (torch.norm(tgt_embedding, dim=1, keepdim=True) + 1e-12)
    return torch.mean(torch.norm(translated - target, dim=1))


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


def test_validation(model, val_loader, cuda):
    model.eval()
    correct3 = 0
    total = len(val_loader.dataset)
    m = nn.Softmax(dim=1)

    with torch.no_grad():
        for val_data, val_label in val_loader:
            if cuda:
                val_data, val_label = val_data.cuda(), val_label.cuda()

            cls_label = val_label[:, 0]
            vib_data = val_data[:, :1024]
            aud_data = val_data[:, 1024:]

            pred3 = model(vib_data, aud_data)
            correct3 += m(pred3).max(1)[1].eq(cls_label).sum().item()

    acc_fusion = 100.0 * correct3 / total
    tqdm.write(f"Val Accuracy    -> Fusion: {acc_fusion:.2f}%")
    return acc_fusion


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
):
    src_iter = iter(src_loader)
    best_val_acc = 0.0
    best_test_acc_at_val = 0.0
    best_per_target_at_val = {}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
    criterion_cls = nn.CrossEntropyLoss()
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
            src_data, src_label = src_data.cuda(), src_label.cuda()

        cls_label = src_label[:, 0]
        vib_data = src_data[:, :1024]
        aud_data = src_data[:, 1024:]

        optimizer.zero_grad()

        pred, vib_emd, aud_emd, vib_proj, aud_proj, _ = model(
            vib_data, aud_data, return_features=True
        )

        cls_loss = criterion_cls(pred, cls_label)

        v2a_loss = compute_translation_loss(vib_emd, aud_emd, model.vib_to_aud)
        a2v_loss = compute_translation_loss(aud_emd, vib_emd, model.aud_to_vib)
        trans_loss = 0.5 * (v2a_loss + a2v_loss)

        emd_proj = torch.stack([vib_proj, aud_proj], dim=1)
        contrast_loss = criterion_contrast(emd_proj, cls_label)

        explore_loss = compute_explore_loss(vib_emd, aud_emd)

        loss = (
            cls_loss
            + alpha_trans * trans_loss
            + alpha_contrast * contrast_loss
            + explore_loss_coeff * explore_loss
        )

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Cls": f"{cls_loss.item():.4f}",
                    "Trans": f"{trans_loss.item():.4f}",
                    "Contrast": f"{contrast_loss.item():.4f}",
                    "Explore": f"{explore_loss.item():.4f}",
                    "BestVal": f"{best_val_acc:.2f}%",
                }
            )

        if i % (log_interval * 20) == 0:
            tqdm.write(f"\n[Iter {i}] Validation & Testing...")
            test_source(model, src_loader, cuda)
            current_val_acc = test_validation(model, val_loader, cuda)
            current_per_target, current_test_acc = evaluate_target_domains(model, target_loaders, cuda)

            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_test_acc_at_val = current_test_acc
                best_per_target_at_val = current_per_target
                torch.save(model.state_dict(), osp.join(CKPT_DIR, f"best_model_{task_name}.pth"))
                tqdm.write(
                    f">>> New Best Val: {best_val_acc:.2f}% | "
                    f"MeanTest@BestVal: {best_test_acc_at_val:.2f}% (Model Saved)"
                )

    return best_val_acc, best_test_acc_at_val, best_per_target_at_val


def test_target(model, test_loader, cuda, title="Target"):
    model.eval()
    correct3 = 0
    total = len(test_loader.dataset)
    m = nn.Softmax(dim=1)

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()

            vib_data = tgt_test_data[:, :1024]
            aud_data = tgt_test_data[:, 1024:]

            tgt_pred3 = model(vib_data, aud_data)
            pred_3 = m(tgt_pred3).max(1)[1]
            correct3 += pred_3.eq(tgt_test_label).sum().item()

    acc_fusion = 100.0 * correct3 / total
    tqdm.write(f"{title:<15} -> Fusion: {acc_fusion:.2f}%")
    return acc_fusion



def evaluate_target_domains(model, target_loaders, cuda):
    per_target_results = {}
    for domain, loader in target_loaders.items():
        per_target_results[int(domain)] = test_target(
            model,
            loader,
            cuda,
            title=f"Target[D{int(domain)}]",
        )

    mean_target_acc = float(np.mean(list(per_target_results.values()))) if per_target_results else 0.0
    per_target_str = ", ".join([f"D{k}:{v:.2f}%" for k, v in per_target_results.items()])
    tqdm.write(f"Target Mean     -> Fusion: {mean_target_acc:.2f}% | PerTarget: {{{per_target_str}}}")
    return per_target_results, mean_target_acc


def test_source(model, test_loader, cuda):
    model.eval()
    correct3 = 0
    total = len(test_loader.dataset)
    m = nn.Softmax(dim=1)

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()

            tgt_test_label = tgt_test_label[:, 0]
            vib_data = tgt_test_data[:, :1024]
            aud_data = tgt_test_data[:, 1024:]

            tgt_pred3 = model(vib_data, aud_data)
            correct3 += m(tgt_pred3).max(1)[1].eq(tgt_test_label).sum().item()

    tqdm.write(f"Source Accuracy -> Fusion: {100.0 * correct3 / total:.2f}%")

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

    model = models.SimMMDG(
        num_classes=class_num,
        embedding_dim=args.embedding_dim,
        cls_hidden_dim=args.hidden_dim,
        trans_hidden_dim=args.trans_hidden_dim,
        proj_hidden_dim=args.hidden_dim,
        proj_out_dim=args.out_dim,
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
