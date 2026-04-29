import argparse
import copy
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
from models import CMRF_Model as models

METHOD_NAME = "CMRF"
DATASET_NAME = "HUST"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 8
log_interval = 10
l2_decay = 5e-4

BASE_DIR = PROJECT_DIR
DATA_DIR = osp.join(BASE_DIR, "data")
ROOT_VIB_PATH = osp.join(DATA_DIR, "Motor_Vib.mat")
ROOT_ASC_PATH = osp.join(DATA_DIR, "Motor_Aud.mat")
RESULT_LOG_DIR = osp.join(BASE_DIR, "outputs", "logs", "CMRF")
CKPT_DIR = osp.join(BASE_DIR, "outputs", "models", "CMRF")


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.dim() != 3:
            raise ValueError("features must be [B, V, D]")

        device = features.device
        batch_size, n_views, _ = features.shape
        features = F.normalize(features, dim=2).reshape(batch_size * n_views, -1)

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
        return loss.sum() / (valid_mask.sum() + 1e-12)


def build_train_val_loaders_from_source(
    src_full_loader, batch_size, kwargs, val_per_class=200, seed=8
):
    features, labels = src_full_loader.dataset.tensors
    labels_np = labels.cpu().numpy()

    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []

    domains = np.unique(labels_np[:, 1])
    classes = np.unique(labels_np[:, 0])

    for domain in domains:
        for cls in classes:
            idx = np.where((labels_np[:, 0] == cls) & (labels_np[:, 1] == domain))[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            val_count = min(val_per_class, len(idx) // 2)
            val_indices.extend(idx[:val_count].tolist())
            train_indices.extend(idx[val_count:].tolist())

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


def mix_feature(modality_1, modality_2, alpha1=1.0, alpha2=1.0):
    batch_size = modality_1.shape[0]
    device = modality_1.device
    dtype = modality_1.dtype

    if alpha1 > 0:
        lam1 = []
        while len(lam1) < batch_size:
            lam = np.random.beta(alpha1, alpha1)
            if lam > 0.5:
                lam1.append(lam)
        lam1 = torch.tensor(lam1, device=device, dtype=dtype).unsqueeze(-1)
    else:
        lam1 = torch.ones(batch_size, 1, device=device, dtype=dtype)

    if alpha2 > 0:
        lam2 = []
        while len(lam2) < batch_size:
            lam = np.random.beta(alpha2, alpha2)
            if lam < 0.5:
                lam2.append(lam)
        lam2 = torch.tensor(lam2, device=device, dtype=dtype).unsqueeze(-1)
    else:
        lam2 = torch.zeros(batch_size, 1, device=device, dtype=dtype)

    mixed_1 = lam1 * modality_1 + (1.0 - lam1) * modality_2
    mixed_2 = lam2 * modality_1 + (1.0 - lam2) * modality_2
    return mixed_1, mixed_2, (lam1, lam2)


def compute_explore_loss(vib_embedding, aud_embedding):
    vib_dim = vib_embedding.size(1) // 2
    aud_dim = aud_embedding.size(1) // 2
    
    v1 = F.normalize(vib_embedding[:, :vib_dim], p=2, dim=1)
    v2 = F.normalize(vib_embedding[:, vib_dim:], p=2, dim=1)
    
    a1 = F.normalize(aud_embedding[:, :aud_dim], p=2, dim=1)
    a2 = F.normalize(aud_embedding[:, aud_dim:], p=2, dim=1)
    
    loss_e = -F.mse_loss(v1, v2) - F.mse_loss(a1, a2)
    
    return loss_e / 2.0


def update_sma_model(student_model, teacher_model, sma_count):
    new_state = {}
    student_state = student_model.state_dict()
    teacher_state = teacher_model.state_dict()

    for name, student_param in student_state.items():
        teacher_param = teacher_state[name]
        new_state[name] = (
            teacher_param.detach().clone() * sma_count + student_param.detach().clone()
        ) / (sma_count + 1.0)

    teacher_model.load_state_dict(new_state)


def forward_model(model, vib_data, aud_data):
    return model(vib_data, aud_data, return_features=True)


def compute_cmrf_losses(
    outputs,
    labels,
    criterion_cls,
    criterion_contrast,
    teacher_outputs=None,
    alpha_contrast=3.0,
    explore_loss_coeff=0.7,
    mix_coef=2.0,
    distill_coef=3.0,
    mix_alpha=1.0,
    use_contrast=True,
    use_distill=True,
):
    fusion_pred = outputs["fusion_pred"]
    vib_pred = outputs["vib_pred"]
    aud_pred = outputs["aud_pred"]
    vib_embedding = outputs["vib_embedding"]
    aud_embedding = outputs["aud_embedding"]
    vib_proj = outputs["vib_proj"]
    aud_proj = outputs["aud_proj"]

    cls_loss = criterion_cls(fusion_pred, labels)
    unimodal_loss = criterion_cls(vib_pred, labels) + 0.5 * criterion_cls(aud_pred, labels)
    explore_loss = compute_explore_loss(vib_embedding, aud_embedding)

    loss = cls_loss + mix_coef * unimodal_loss + explore_loss_coeff * explore_loss

    contrast_loss = torch.tensor(0.0, device=labels.device)
    if use_contrast:
        contrast_loss = criterion_contrast(torch.stack([vib_proj, aud_proj], dim=1), labels)
        loss = loss + alpha_contrast * contrast_loss

    distill_loss = torch.tensor(0.0, device=labels.device)
    if use_distill and teacher_outputs is not None:
        tea_vib_proj = teacher_outputs["vib_proj"]
        tea_aud_proj = teacher_outputs["aud_proj"]
        vib_mix_teacher, aud_mix_teacher, _ = mix_feature(
            tea_vib_proj, tea_aud_proj, mix_alpha, mix_alpha
        )

        vib_teacher = F.normalize(vib_mix_teacher, dim=1)
        vib_student = F.normalize(vib_proj, dim=1)
        aud_teacher = F.normalize(aud_mix_teacher, dim=1)
        aud_student = F.normalize(aud_proj, dim=1)

        distill_loss = torch.mean(torch.norm(vib_teacher.detach() - vib_student, dim=1))
        distill_loss = distill_loss + 0.5 * torch.mean(
            torch.norm(aud_teacher.detach() - aud_student, dim=1)
        )
        loss = loss + distill_coef * distill_loss

    metrics = {
        "loss": loss,
        "cls_loss": cls_loss.detach(),
        "unimodal_loss": unimodal_loss.detach(),
        "contrast_loss": contrast_loss.detach(),
        "explore_loss": explore_loss.detach(),
        "distill_loss": distill_loss.detach(),
        "fusion_pred": fusion_pred,
    }
    return metrics


def test_validation(model, val_loader, cuda):
    model.eval()
    correct = 0
    total = len(val_loader.dataset)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for val_data, val_label in val_loader:
            if cuda:
                val_data, val_label = val_data.cuda(), val_label.cuda()

            cls_label = val_label[:, 0]
            vib_data = val_data[:, :1024]
            aud_data = val_data[:, 1024:]

            pred = model(vib_data, aud_data)
            correct += softmax(pred).max(1)[1].eq(cls_label).sum().item()

    acc_fusion = 100.0 * correct / total
    tqdm.write(f"Val Accuracy    -> Fusion: {acc_fusion:.2f}%")
    return acc_fusion


def test_target(model, test_loader, cuda, title="Target"):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for test_data, test_label in test_loader:
            if cuda:
                test_data, test_label = test_data.cuda(), test_label.cuda()

            vib_data = test_data[:, :1024]
            aud_data = test_data[:, 1024:]
            pred = model(vib_data, aud_data)
            correct += softmax(pred).max(1)[1].eq(test_label).sum().item()

    acc_fusion = 100.0 * correct / total
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
    correct = 0
    total = len(test_loader.dataset)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for test_data, test_label in test_loader:
            if cuda:
                test_data, test_label = test_data.cuda(), test_label.cuda()

            cls_label = test_label[:, 0]
            vib_data = test_data[:, :1024]
            aud_data = test_data[:, 1024:]

            pred = model(vib_data, aud_data)
            correct += softmax(pred).max(1)[1].eq(cls_label).sum().item()

    tqdm.write(f"Source Accuracy -> Fusion: {100.0 * correct / total:.2f}%")


def train(
    model,
    src_loader,
    val_loader,
    target_loaders,
    iteration,
    lr,
    cuda,
    task_name,
    alpha_contrast=3.0,
    temp=0.1,
    explore_loss_coeff=0.7,
    mix_alpha=1.0,
    mix_coef=2.0,
    distill_coef=3.0,
    sma_start_step=400,
    use_cm_mixup=True,
    use_contrast=True,
    use_distill=True,
    use_sma=True,
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

    teacher_model = None
    sma_count = 0
    if use_sma:
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()

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

        outputs = forward_model(model, vib_data, aud_data)

        teacher_outputs = None
        if use_cm_mixup and use_sma and teacher_model is not None and i > sma_start_step:
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = forward_model(teacher_model, vib_data, aud_data)

        metrics = compute_cmrf_losses(
            outputs,
            cls_label,
            criterion_cls,
            criterion_contrast,
            teacher_outputs=teacher_outputs,
            alpha_contrast=alpha_contrast,
            explore_loss_coeff=explore_loss_coeff,
            mix_coef=mix_coef if use_cm_mixup else 0.0,
            distill_coef=distill_coef,
            mix_alpha=mix_alpha,
            use_contrast=use_contrast and use_cm_mixup,
            use_distill=use_distill and use_cm_mixup and use_sma and i > sma_start_step,
        )

        loss = metrics["loss"]
        loss.backward()
        optimizer.step()

        if use_sma and teacher_model is not None:
            if i > sma_start_step:
                sma_count += 1
                update_sma_model(model, teacher_model, sma_count)
            else:
                teacher_model.load_state_dict(copy.deepcopy(model.state_dict()))

        if i % log_interval == 0:
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Cls": f"{metrics['cls_loss'].item():.4f}",
                    "Uni": f"{metrics['unimodal_loss'].item():.4f}",
                    "Ctr": f"{metrics['contrast_loss'].item():.4f}",
                    "Exp": f"{metrics['explore_loss'].item():.4f}",
                    "Dis": f"{metrics['distill_loss'].item():.4f}",
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
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--alpha_contrast", type=float, default=3.0)
    parser.add_argument("--explore_loss_coeff", type=float, default=0.7)
    parser.add_argument("--mix_alpha", type=float, default=1.0)
    parser.add_argument("--mix_coef", type=float, default=2.0)
    parser.add_argument("--distill_coef", type=float, default=3.0)
    parser.add_argument("--sma_start_step", type=int, default=100)
    parser.add_argument("--cm_mixup", dest="cm_mixup", action="store_true")
    parser.add_argument("--contrast", dest="contrast", action="store_true")
    parser.add_argument("--distill", dest="distill", action="store_true")
    parser.add_argument("--sma", dest="sma", action="store_true")
    parser.set_defaults(cm_mixup=True, contrast=True, distill=True, sma=True)
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

    model = models.CMRF(
        num_classes=class_num,
        embedding_dim=args.embedding_dim,
        cls_hidden_dim=args.hidden_dim,
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
        alpha_contrast=args.alpha_contrast,
        temp=args.temp,
        explore_loss_coeff=args.explore_loss_coeff,
        mix_alpha=args.mix_alpha,
        mix_coef=args.mix_coef,
        distill_coef=args.distill_coef,
        sma_start_step=args.sma_start_step,
        use_cm_mixup=args.cm_mixup,
        use_contrast=args.contrast,
        use_distill=args.distill,
        use_sma=args.sma,
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
