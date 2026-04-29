import os
import os.path as osp
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
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
from models import GMP_Model as models

METHOD_NAME = "GMP"
DATASET_NAME = "HUST"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 8
log_interval = 10
l2_decay = 5e-4


ALPHA_REV2 = 0.3
ALPHA_K = 0.5
ALPHA_P = 0.5
LAMBDA_DOMAIN_LOCAL = 0.5
LAMBDA_CLS_LOCAL = 3.0

BASE_DIR = PROJECT_DIR
DATA_DIR = osp.join(BASE_DIR, 'data')
ROOT_VIB_PATH = osp.join(DATA_DIR, 'Motor_Vib.mat')
ROOT_ASC_PATH = osp.join(DATA_DIR, 'Motor_Aud.mat')
RESULT_LOG_DIR = osp.join(BASE_DIR, 'outputs', 'logs', 'GMP')


def build_train_val_loaders_from_source(src_full_loader, batch_size, kwargs, val_per_class=200, seed=8):
    features, labels = src_full_loader.dataset.tensors
    labels_np = labels.cpu().numpy()  # [N, 2] -> [class, domain]

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
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs
    )

    print(f"Split source train/val done: train={len(train_set)}, val={len(val_set)}, val_per_class={val_per_class}")
    return train_loader, val_loader


def _mean_confidence(logits, labels):
    probs = torch.softmax(logits, dim=1)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    gt_probs = probs[batch_idx, labels]
    return gt_probs.mean()


def _compute_diff_ratios(conf_dict, eps=1e-6):
    ratios = {}
    keys = list(conf_dict.keys())
    if len(keys) <= 1:
        for k in keys:
            ratios[k] = torch.tensor(1.0, device=conf_dict[k].device)
        return ratios

    for k in keys:
        others = [conf_dict[j] for j in keys if j != k]
        others_mean = torch.stack(others).mean()
        ratios[k] = conf_dict[k] / (others_mean + eps)
    return ratios


def _compute_diff_ratios_domain(conf_dict, eps=1e-6):
    ratios = {}
    keys = list(conf_dict.keys())
    if len(keys) <= 1:
        for k in keys:
            ratios[k] = torch.tensor(1.0, device=conf_dict[k].device)
        return ratios

    for k in keys:
        others = [conf_dict[j] for j in keys if j != k]
        others_mean = torch.stack(others).mean()
        ratios[k] = others_mean / (conf_dict[k] + eps)
    return ratios


def _mod_coeff_from_ratio_tanh(ratio, alpha):
    suppress_val = 1.0 - torch.tanh(alpha * ratio)
    suppress_val = torch.clamp(suppress_val, min=0.0, max=1.0)
    return torch.where(ratio > 1.0, suppress_val, torch.ones_like(ratio))


def _dot_grads(grad_list_a, grad_list_b):
    dot = None
    device = None
    with torch.no_grad():
        for ga, gb in zip(grad_list_a, grad_list_b):
            if ga is None or gb is None:
                continue
            if device is None:
                device = ga.device
            v = (ga * gb).sum()
            dot = v if dot is None else (dot + v)
    if dot is None:
        return torch.tensor(0.0, device=device if device is not None else torch.device('cpu'))
    return dot


def _norm_sq_grads(grad_list, eps=1e-12):
    s = None
    device = None
    with torch.no_grad():
        for g in grad_list:
            if g is None:
                continue
            if device is None:
                device = g.device
            v = (g * g).sum()
            s = v if s is None else (s + v)
    if s is None:
        return torch.tensor(eps, device=device if device is not None else torch.device('cpu'))
    return s + eps


def _project_conflict(g_strong, g_weak, eps=1e-12):
    dot_sw = _dot_grads(g_strong, g_weak)
    weak_norm_sq = _norm_sq_grads(g_weak, eps=eps)
    coeff = dot_sw / weak_norm_sq
    out = []
    for gs, gw in zip(g_strong, g_weak):
        if gs is None:
            out.append(None)
        elif gw is None:
            out.append(gs)
        else:
            out.append(gs - coeff * gw)
    return out


def _scale_grads(grads, scale):
    return [None if g is None else (g * scale) for g in grads]


def _add_grads(ga, gb):
    out = []
    for a, b in zip(ga, gb):
        if a is None and b is None:
            out.append(None)
        elif a is None:
            out.append(b)
        elif b is None:
            out.append(a)
        else:
            out.append(a + b)
    return out


def train_one_step(model, vib_data, aud_data, cls_label, domain_label, criterion, optimizer,
                   vib_modal_params, aud_modal_params, use_domain_adv=True):
    outputs = model.forward_train(vib_data, aud_data, alpha_domain=ALPHA_REV2)

    modal_outputs = {
        'v': {
            'cls_logits': outputs['v_logit'],
            'domain_logits': outputs['v_domain_logit'],
            'cls_loss': criterion(outputs['v_logit'], cls_label),
            'domain_loss': criterion(outputs['v_domain_logit'], domain_label) if use_domain_adv else None,
            'params': vib_modal_params,
        },
        'a': {
            'cls_logits': outputs['a_logit'],
            'domain_logits': outputs['a_domain_logit'],
            'cls_loss': criterion(outputs['a_logit'], cls_label),
            'domain_loss': criterion(outputs['a_domain_logit'], domain_label) if use_domain_adv else None,
            'params': aud_modal_params,
        },
    }

    sem_conf = {m: _mean_confidence(modal_outputs[m]['cls_logits'], cls_label) for m in modal_outputs}
    sem_ratio = _compute_diff_ratios(sem_conf)
    sem_coeff = {m: _mod_coeff_from_ratio_tanh(sem_ratio[m], ALPHA_K) for m in modal_outputs}

    if use_domain_adv:
        dom_conf = {m: _mean_confidence(modal_outputs[m]['domain_logits'], domain_label) for m in modal_outputs}
        dom_ratio = _compute_diff_ratios_domain(dom_conf)
        dom_coeff = {m: _mod_coeff_from_ratio_tanh(dom_ratio[m], ALPHA_P) for m in modal_outputs}

    final_modal_grads = {}
    for m in modal_outputs:
        params_m = modal_outputs[m]['params']
        g_cls = torch.autograd.grad(
            modal_outputs[m]['cls_loss'] * LAMBDA_CLS_LOCAL,
            params_m,
            retain_graph=True,
            allow_unused=True,
        )

        g_cls = _scale_grads(g_cls, sem_coeff[m])

        if use_domain_adv:
            g_dom = torch.autograd.grad(
                modal_outputs[m]['domain_loss'] * LAMBDA_DOMAIN_LOCAL,
                params_m,
                retain_graph=True,
                allow_unused=True,
            )
            g_dom = _scale_grads(g_dom, dom_coeff[m])

            if _dot_grads(g_cls, g_dom) < 0:
                if sem_ratio[m] >= dom_ratio[m]:
                    g_cls = _project_conflict(g_cls, g_dom)
                else:
                    g_dom = _project_conflict(g_dom, g_cls)

            final_modal_grads[m] = _add_grads(g_cls, g_dom)
        else:
            final_modal_grads[m] = g_cls

    fusion_logits = outputs['fusion_logit']
    fusion_loss = criterion(fusion_logits, cls_label)

    optimizer.zero_grad()
    fusion_loss.backward()

    for m in modal_outputs:
        for p, g in zip(modal_outputs[m]['params'], final_modal_grads[m]):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())

    optimizer.step()

    cls_loss_local = 0.5 * (modal_outputs['v']['cls_loss'] + modal_outputs['a']['cls_loss'])
    if use_domain_adv:
        domain_loss_local = 0.5 * (modal_outputs['v']['domain_loss'] + modal_outputs['a']['domain_loss'])
        display_loss = fusion_loss + LAMBDA_CLS_LOCAL * cls_loss_local + LAMBDA_DOMAIN_LOCAL * domain_loss_local
    else:
        display_loss = fusion_loss + LAMBDA_CLS_LOCAL * cls_loss_local
    return fusion_logits, display_loss


def evaluate(model, data_loader, cuda, with_domain_label=False, title='Eval'):
    model.eval()
    correct = 0
    total = len(data_loader.dataset)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, label in data_loader:
            if cuda:
                data, label = data.cuda(), label.cuda()

            cls_label = label[:, 0] if with_domain_label else label
            vib_data = data[:, :1024]
            aud_data = data[:, 1024:]

            pred = model(vib_data, aud_data)
            loss = criterion(pred, cls_label)

            total_loss += loss.item() * pred.size(0)
            pred_label = pred.argmax(dim=1)
            correct += pred_label.eq(cls_label).sum().item()

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    tqdm.write(f"{title} -> Fusion Acc: {acc:.2f}%, Loss: {avg_loss:.4f}")
    return acc, avg_loss


def evaluate_target_domains(model, target_loaders, cuda):
    per_target_results = {}
    for domain, loader in target_loaders.items():
        single_acc, _ = evaluate(
            model,
            loader,
            cuda,
            with_domain_label=False,
            title=f"Target[D{int(domain)}]",
        )
        per_target_results[int(domain)] = single_acc

    mean_target_acc = float(np.mean(list(per_target_results.values()))) if per_target_results else 0.0
    per_target_str = ', '.join([f"D{k}:{v:.2f}%" for k, v in per_target_results.items()])
    tqdm.write(f"Target Mean -> Fusion Acc: {mean_target_acc:.2f}% | PerTarget: {{{per_target_str}}}")
    return per_target_results, mean_target_acc


def train(model, src_loader, val_loader, target_loaders, iteration, lr, cuda, task_name, use_domain_adv=True):
    src_iter = iter(src_loader)
    best_val_acc = 0.0
    best_test_acc_at_val = 0.0
    best_per_target_at_val = {}
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)

    vib_modal_params = list(model.vib_net.parameters()) + list(model.cls_v.parameters())
    aud_modal_params = list(model.aud_net.parameters()) + list(model.cls_a.parameters())

    pbar = tqdm(range(1, iteration + 1), desc=f"Training {task_name}", unit="iter")

    for i in pbar:
        model.train()
        try:
            src_data, src_label = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            src_data, src_label = next(src_iter)

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

        cls_label = src_label[:, 0]
        domain_label = src_label[:, 1] if use_domain_adv else None
        vib_data = src_data[:, :1024]
        aud_data = src_data[:, 1024:]

        _, loss = train_one_step(
            model,
            vib_data,
            aud_data,
            cls_label,
            domain_label,
            criterion,
            optimizer,
            vib_modal_params,
            aud_modal_params,
            use_domain_adv=use_domain_adv,
        )

        if i % log_interval == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'BestVal': f'{best_val_acc:.2f}%'})

        if i % (log_interval * 20) == 0:
            tqdm.write(f"\n[Iter {i}] Validation & Testing...")
            evaluate(model, src_loader, cuda, with_domain_label=True, title='Source')
            current_val_acc, _ = evaluate(model, val_loader, cuda, with_domain_label=True, title='Val')
            current_per_target, current_test_acc = evaluate_target_domains(model, target_loaders, cuda)

            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_test_acc_at_val = current_test_acc
                best_per_target_at_val = current_per_target
                torch.save(model.state_dict(), osp.join(CKPT_DIR, f'best_model_{task_name}.pth'))
                best_per_target_str = ', '.join([f"D{k}:{v:.2f}%" for k, v in best_per_target_at_val.items()])
                tqdm.write(
                    f">>> New Best Val: {best_val_acc:.2f}% | "
                    f"MeanTest@BestVal: {best_test_acc_at_val:.2f}% | "
                    f"PerTarget@BestVal: {{{best_per_target_str}}} (Model Saved)"
                )

    return best_val_acc, best_test_acc_at_val, best_per_target_at_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--class_num', type=int, default=6)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--l2_decay', type=float, default=5e-4)
    parser.add_argument('--val_per_class', type=int, default=200)
    parser.add_argument('--alpha_rev2', type=float, default=0.3)
    parser.add_argument('--alpha_k', type=float, default=0.5)
    parser.add_argument('--alpha_p', type=float, default=0.5)
    parser.add_argument('--lambda_domain_local', type=float, default=0.5)
    parser.add_argument('--lambda_cls_local', type=float, default=3.0)
    parser.add_argument('-s', '--source_domain', nargs='+', required=True,
                        help='Source domain(s), e.g. -s D1 D2 D3 or -s 1 2 3')
    parser.add_argument('-t', '--target_domain', nargs='+', required=True,
                        help='Target domain(s), e.g. -t D4 or -t 4')
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()

    iteration = args.iteration
    batch_size = args.batch_size
    lr = args.lr
    class_num = args.class_num
    seed = args.seed
    log_interval = args.log_interval
    l2_decay = args.l2_decay
    ALPHA_REV2 = args.alpha_rev2
    ALPHA_K = args.alpha_k
    ALPHA_P = args.alpha_p
    LAMBDA_DOMAIN_LOCAL = args.lambda_domain_local
    LAMBDA_CLS_LOCAL = args.lambda_cls_local

    source_domains, target_domains = parse_hust_domain_args(args.source_domain, args.target_domain)
    sourcelist = np.array(source_domains)
    target = np.array(target_domains)
    task_name = build_task_name(sourcelist, target)
    dg_mode = infer_dg_mode(sourcelist)
    run_name = build_run_name(METHOD_NAME, DATASET_NAME, sourcelist, target, seed, args.run_name)
    RESULT_LOG_DIR, CKPT_DIR, log_path = build_output_paths(
        __file__, DATASET_NAME, METHOD_NAME, dg_mode, run_name
    )

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    write_run_header(log_path, args, METHOD_NAME, dg_mode, sourcelist, target)

    print(f"\n>>> Running {task_name}: {sourcelist} -> {target}")
    print(f"Log path: {log_path}")

    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    src_full_loader = data_loader_1d.load_training(
        sourcelist, False, batch_size, kwargs,
        root_vib=ROOT_VIB_PATH, root_asc=ROOT_ASC_PATH, class_num=class_num
    )
    src_loader, val_loader = build_train_val_loaders_from_source(
        src_full_loader,
        batch_size=batch_size,
        kwargs=kwargs,
        val_per_class=args.val_per_class,
        seed=seed
    )
    target_loaders = {}
    for d in target.tolist():
        target_loaders[int(d)] = data_loader_1d.load_testing(
            np.array([int(d)]), False, batch_size, kwargs,
            root_vib=ROOT_VIB_PATH, root_asc=ROOT_ASC_PATH, class_num=class_num
        )

    model = models.GMP(num_classes=class_num, num_domains=len(sourcelist))
    if cuda:
        model.cuda()

    use_domain_adv = len(sourcelist) > 1

    best_val_acc, test_at_best_val, best_per_target_at_val = train(
        model, src_loader, val_loader, target_loaders, iteration, lr, cuda, task_name,
        use_domain_adv=use_domain_adv,
    )

    best_ckpt_path = osp.join(CKPT_DIR, f'best_model_{task_name}.pth')
    if osp.exists(best_ckpt_path):
        state_dict = torch.load(best_ckpt_path, map_location='cuda' if cuda else 'cpu')
        model.load_state_dict(state_dict)

    per_target_results, per_target_mean = evaluate_target_domains(model, target_loaders, cuda)
    if not best_per_target_at_val:
        best_per_target_at_val = per_target_results
        test_at_best_val = per_target_mean

    per_target_str = ', '.join([f"{k}:{v:.2f}%" for k, v in per_target_results.items()])
    best_per_target_str = ', '.join([f"{k}:{v:.2f}%" for k, v in best_per_target_at_val.items()])

    log_entry = (
        f"{task_name}: Source={sourcelist.tolist()}, Target={target.tolist()}, "
        f"Best Val(Fusion)={best_val_acc:.2f}%, Test@BestVal(Fusion)={test_at_best_val:.2f}%, "
        f"BestPerTarget(Fusion)={{{best_per_target_str}}}, "
        f"PerTarget(Fusion)={{{per_target_str}}}, MeanTarget(Fusion)={per_target_mean:.2f}%"
    )
    print(log_entry)
    append_best_result(log_path, "", best_val_acc, test_at_best_val, best_per_target_at_val, test_at_best_val)

    print(f"Finished {task_name}. Result saved to {log_path}")
