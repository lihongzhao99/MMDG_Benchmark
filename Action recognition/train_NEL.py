"""
NEL Training Script — EPIC-Kitchens
Faithful reproduction of:
  "Nonpolarized embedding learning in multimodal domain generalization"
  Neurocomputing 658 (2025) 131754

Overall loss (Eq. 14):
    L = cls_loss + α * L_UNC + (1 - α) * L_C + β * L_NEE

where
    L_UNC : unsupervised contrastive on mean-shift embeddings (Eq. 8)
    L_C   : supervised contrastive on initial embeddings       (Eq. 9)
    L_NEE : Von Neumann entropy of autocorrelation matrix      (Eq. 12)
"""

import os
import sys
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import cv2

cpu_num = 1
os.environ["OMP_NUM_THREADS"]          = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"]     = str(cpu_num)
os.environ["MKL_NUM_THREADS"]          = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"]   = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"]      = str(cpu_num)
torch.set_num_threads(cpu_num)
cv2.setNumThreads(cpu_num)
cv2.ocl.setUseOpenCL(False)

import argparse
import time
import tqdm
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from mmaction.apis import init_recognizer
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments

from dataloaders.dataloader_EPIC import EPICDOMAIN
from dataloaders.dataloader_HAC import HACDOMAIN


METHOD_NAME = 'NEL'


def configure_dataset_args(args):
    args.dataset_name = 'HAC' if args.dataset == 'hac' else 'EPIC'
    return HACDOMAIN if args.dataset == 'hac' else EPICDOMAIN


def make_dataset_kwargs(args, split, domain, cfg=None, cfg_flow=None, source=True):
    kwargs = dict(
        split=split,
        domain=domain,
        cfg=cfg,
        cfg_flow=cfg_flow,
        datapath=args.datapath,
        use_video=args.use_video,
        use_flow=args.use_flow,
        use_audio=args.use_audio,
    )
    if args.dataset == 'hac':
        kwargs['source'] = source
    return kwargs


class SupConLoss(nn.Module):
    """Supervised contrastive loss for per-modality projected embeddings."""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = mask.sum(1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class ConMeanShiftLoss(nn.Module):
    """Unsupervised contrastive loss between mean-shift embeddings of two views."""

    def __init__(self, n_views=2, alpha=0.5, temperature=0.25):
        super().__init__()
        self.n_views = n_views
        self.alpha = alpha
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
        log_prob = pos_sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + exp_pos + 1e-12)
        return -log_prob.mean()


class DOMAINNVIEWS(torch.utils.data.Dataset):
    def __init__(self, base_dataset, *args, **kwargs):
        self.view1 = base_dataset(*args, **kwargs)
        self.view2 = base_dataset(*args, **kwargs)

    def __getitem__(self, index):
        clip1, flow1, spectrogram1, label1 = self.view1[index]
        clip2, flow2, spectrogram2, label2 = self.view2[index]
        if label1 != label2:
            raise ValueError(f"Two {globals()['args'].dataset_name} views have different labels at index {index}: {label1} != {label2}")
        return (clip1, clip2), (flow1, flow2), (spectrogram1, spectrogram2), label1

    def __len__(self):
        return len(self.view1)


def _normalize_domains(domain_arg):
    if domain_arg is None:
        return []
    if isinstance(domain_arg, (list, tuple)):
        return list(domain_arg)
    return [domain_arg]


def _infer_dg_mode(source_domains):
    return 'single_source_dg' if len(source_domains) == 1 else 'multi_source_dg'


def _build_modality_code(use_video, use_audio, use_flow):
    return '_'.join(
        key for key, enabled in [('v', use_video), ('a', use_audio), ('f', use_flow)]
        if enabled
    )


def _format_domain_tag(domains):
    return "-".join(str(domain) for domain in domains)


def build_run_name(method_name, dataset_name, source_domains, target_domains,
                   modal_code, seed, appen="", run_name=None):
    if run_name:
        return run_name
    appen_tag = f"_{appen}" if appen else ""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{method_name}_{dataset_name}_"
        f"{_format_domain_tag(source_domains)}_to_{_format_domain_tag(target_domains)}_"
        f"{modal_code}_seed{seed}{appen_tag}_{run_id}"
    )


def build_output_paths(script_file, dataset_name, method_name, dg_mode, run_name):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    output_root = os.path.join(script_dir, "outputs")
    log_dir = os.path.join(output_root, "logs", dataset_name, method_name, dg_mode)
    model_dir = os.path.join(output_root, "models", dataset_name, method_name, dg_mode)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir, os.path.join(log_dir, run_name + ".csv")


def compute_mean_shift_embedding(embeddings, k=8):
    """
    Compute mean-shift embeddings using k-nearest neighbors.
    Implements Eq. (4)–(6) of the paper.

    Kernel weight φ(e):
        0.5       if e == e_i  (anchor)
        0.5 / k   otherwise    (each of the k neighbors)

    The result m_i is L2-normalised onto the unit hypersphere (Eq. 6).

    Args:
        embeddings : [N, D]  initial embeddings (already L2-normalised)
        k          : number of nearest neighbours
    Returns:
        mean_shift_embeddings : [N, D]  normalised mean-shift embeddings
    """
    N, D = embeddings.shape

    embeddings_norm = F.normalize(embeddings, dim=1)
    if N < 2:
        return embeddings_norm
    k = min(k, N - 1)
    sim = torch.matmul(embeddings_norm, embeddings_norm.t())


    _, knn_idx = torch.topk(sim, k=k + 1, dim=1, largest=True)
    knn_idx = knn_idx[:, 1:]


    neighbor_weight = 0.5 / k
    neighbors = embeddings[knn_idx]
    weighted  = 0.5 * embeddings + neighbor_weight * neighbors.sum(dim=1)

    return F.normalize(weighted, dim=1)




def compute_nee_loss(embeddings):
    """
    Nonpolarized Embedding Estimate loss  (L_NEE, Eq. 11–12).

    Autocorrelation matrix (no centring, as defined in the paper):
        C_auto = E^T E / n          (Eq. 11)

    Loss (minimise to maximise entropy, i.e. push eigenvalues to be uniform):
        L_NEE = Σ_i  λ_i · log(λ_i)   (Eq. 12)

    Args:
        embeddings : [N, D]  fused multimodal embeddings
    Returns:
        scalar loss
    """
    N, D = embeddings.shape


    C_auto = torch.matmul(embeddings.t(), embeddings) / N

    try:
        eigenvalues = torch.linalg.eigvalsh(C_auto)
        eigenvalues = eigenvalues[eigenvalues > 1e-6]
        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)


        loss = (eigenvalues * torch.log(eigenvalues + 1e-10)).sum()
    except Exception:
        loss = torch.tensor(0.0, device=embeddings.device)

    return loss


def _build_fused_feat(v_emd=None, a_emd=None, f_emd=None):
    feats = []
    if v_emd is not None:
        feats.append(v_emd)
    if a_emd is not None:
        feats.append(a_emd)
    if f_emd is not None:
        feats.append(f_emd)
    if not feats:
        raise ValueError('At least one modality must be enabled.')
    return torch.cat(feats, dim=1)


def _build_projected_views(v_emd=None, a_emd=None, f_emd=None):
    proj_list = []
    if v_emd is not None:
        proj_list.append(v_proj(v_emd[:, :v_emd.shape[1] // 2]))
    if a_emd is not None:
        proj_list.append(a_proj(a_emd[:, :a_emd.shape[1] // 2]))
    if f_emd is not None:
        proj_list.append(f_proj(f_emd[:, :f_emd.shape[1] // 2]))
    if not proj_list:
        raise ValueError('At least one modality must be enabled.')
    return torch.stack(proj_list, dim=1)


def train_one_step(clip, labels, flow, spectrogram):
    """
    One training step implementing the full NEL objective (Eq. 14):

        L = cls_loss + α * L_UNC + (1 - α) * L_C + β * L_NEE
    """
    labels = torch.cat([labels.cuda(), labels.cuda()], dim=0)
    optim.zero_grad()

    if args.use_video:
        clip = torch.cat([clip[0]['imgs'].cuda().squeeze(1),
                          clip[1]['imgs'].cuda().squeeze(1)], dim=0)
    if args.use_flow:
        flow = torch.cat([flow[0]['imgs'].cuda().squeeze(1),
                          flow[1]['imgs'].cuda().squeeze(1)], dim=0)
    if args.use_audio:
        spectrogram = torch.cat([spectrogram[0].unsqueeze(1).cuda(),
                                  spectrogram[1].unsqueeze(1).cuda()], dim=0)


    with torch.no_grad():
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = (x_slow.detach(), x_fast.detach())
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)


    v_emd = audio_emd = f_emd = None

    if args.use_video:
        v_feat   = model.module.backbone.get_predict(v_feat)
        _, v_emd = model.module.cls_head(v_feat)

    if args.use_flow:
        f_feat   = model_flow.module.backbone.get_predict(f_feat.detach())
        _, f_emd = model_flow.module.cls_head(f_feat)

    if args.use_audio:
        _, audio_emd = audio_cls_model(audio_feat.detach())


    feat = _build_fused_feat(v_emd=v_emd, a_emd=audio_emd, f_emd=f_emd)

    predict  = mlp_cls(feat)
    cls_loss = criterion(predict, labels)
    predict, _ = predict.chunk(2)


    emd_proj = _build_projected_views(v_emd=v_emd, a_emd=audio_emd, f_emd=f_emd)

    loss_supervised = criterion_supcon(emd_proj, labels)


    N_half = emd_proj.shape[0] // 2
    n_mod  = emd_proj.shape[1]

    ms_projs = []
    for m in range(n_mod):
        ms = compute_mean_shift_embedding(emd_proj[:, m, :], k=args.k)
        ms_projs.append(ms)
    ms_projs = torch.stack(ms_projs, dim=1)

    ms_view1 = ms_projs[:N_half].reshape(N_half, -1)
    ms_view2 = ms_projs[N_half:].reshape(N_half, -1)

    loss_unsupervised = criterion_unsupcon(ms_view1, ms_view2)


    loss_nee = compute_nee_loss(feat)



    loss = (cls_loss
            + args.alpha * loss_unsupervised
            + (1.0 - args.alpha) * loss_supervised
            + args.beta * loss_nee)

    loss = loss / 2
    loss.backward()
    optim.step()

    return predict, loss




def validate_one_step(clip, labels, flow, spectrogram):
    labels = torch.cat([labels.cuda(), labels.cuda()], dim=0)

    with torch.no_grad():
        v_emd = audio_emd = f_emd = None

        if args.use_video:
            clip = torch.cat([clip[0]['imgs'].cuda().squeeze(1),
                              clip[1]['imgs'].cuda().squeeze(1)], dim=0)
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = model.module.backbone.get_predict((x_slow, x_fast))
            _, v_emd = model.module.cls_head(v_feat)

        if args.use_audio:
            spectrogram = torch.cat([spectrogram[0].unsqueeze(1).cuda(),
                                      spectrogram[1].unsqueeze(1).cuda()], dim=0)
            _, audio_feat, _ = audio_model(spectrogram)
            _, audio_emd = audio_cls_model(audio_feat)

        if args.use_flow:
            flow = torch.cat([flow[0]['imgs'].cuda().squeeze(1),
                              flow[1]['imgs'].cuda().squeeze(1)], dim=0)
            f_feat = model_flow.module.backbone.get_feature(flow)
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            _, f_emd = model_flow.module.cls_head(f_feat)

        feat = _build_fused_feat(v_emd=v_emd, a_emd=audio_emd, f_emd=f_emd)

        predict = mlp_cls(feat)

    loss = criterion(predict, labels)
    predict, _ = predict.chunk(2)
    return predict, loss




class Encoder(nn.Module):
    """MLP classifier for fused multimodal features."""
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)


class ProjectHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps per-modality embeddings to the contrastive space.
    """
    def __init__(self, input_dim=1152, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat):
        return F.normalize(self.head(feat), dim=1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--dataset', type=str, required=True, choices=['epic', 'hac'])
    parser.add_argument('-s', '--source_domain', nargs='+', required=True)
    parser.add_argument('-t', '--target_domain', nargs='+', required=True)
    parser.add_argument('--datapath',       type=str,
                        default='/path/to/DATA_ROOT')
    parser.add_argument('--num_class', type=int, required=True)


    parser.add_argument('--lr',      type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--bsz',     type=int,   default=16)
    parser.add_argument('--nepochs', type=int,   default=15)
    parser.add_argument('--seed',    type=int,   default=0)



    parser.add_argument('--alpha', type=float, default=0.7,
                        help='weight for L_UNC; (1-alpha) weights L_C  (Eq.14)')

    parser.add_argument('--beta',  type=float, default=None,
                        help='weight for L_NEE; defaults to 1/bsz  (Eq.14)')

    parser.add_argument('--temp_s', type=float, default=0.1,
                        help='temperature τ_s for supervised contrastive (L_C)')
    parser.add_argument('--temp_u', type=float, default=0.25,
                        help='temperature τ_u for unsupervised contrastive (L_UNC)')

    parser.add_argument('--k', type=int, default=8,
                        help='k for k-NN in mean-shift embedding (Section 3.3)')


    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--out_dim',    type=int, default=128)


    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow',  action='store_true')


    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best',       action='store_true')
    parser.add_argument('--resumef',         action='store_true')
    parser.add_argument('--BestEpoch',       type=int,   default=0)
    parser.add_argument('--BestAcc',         type=float, default=0)
    parser.add_argument('--BestTestAcc',     type=float, default=0)
    parser.add_argument('--appen',           type=str,   default='')
    parser.add_argument('--run_name',        type=str,   default='')

    args = parser.parse_args()
    DatasetClass = configure_dataset_args(args)
    DATASET_NAME = args.dataset_name

    args.source_domain = _normalize_domains(args.source_domain)
    args.target_domain = _normalize_domains(args.target_domain)
    if len(args.source_domain) < 1:
        raise ValueError('At least one source domain is required.')
    if len(args.target_domain) < 1:
        raise ValueError('At least one target domain is required.')
    args.dg_mode = _infer_dg_mode(args.source_domain)
    print(f'DG mode: {args.dg_mode}')


    if args.beta is None:
        args.beta = 1.0 / args.bsz


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True


    config_file          = os.path.join(SCRIPT_DIR, 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py')
    checkpoint_file      = os.path.join(SCRIPT_DIR, 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth')
    config_file_flow     = os.path.join(SCRIPT_DIR, 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py')
    checkpoint_file_flow = os.path.join(SCRIPT_DIR, 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth')

    device    = torch.device('cuda:0')
    input_dim = 0
    cfg = cfg_flow = None


    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file,
                                device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, args.num_class).cuda()
        cfg = model.cfg
        model = nn.DataParallel(model)
        v_proj = ProjectHead(input_dim=1152,
                             hidden_dim=args.hidden_dim,
                             out_dim=args.out_dim).cuda()
        input_dim += 2304


    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow,
                                     device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, args.num_class).cuda()
        cfg_flow = model_flow.cfg
        model_flow = nn.DataParallel(model_flow)
        f_proj = ProjectHead(input_dim=1024,
                             hidden_dim=args.hidden_dim,
                             out_dim=args.out_dim).cuda()
        input_dim += 2048


    if args.use_audio:
        audio_args  = get_arguments()
        audio_model = AVENet(audio_args)
        ckpt = torch.load(os.path.join(SCRIPT_DIR, 'pretrained_models/vggsound_avgpool.pth.tar'))
        audio_model.load_state_dict(ckpt['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, args.num_class)
        audio_cls_model = audio_cls_model.cuda()

        a_proj = ProjectHead(input_dim=256,
                             hidden_dim=args.hidden_dim,
                             out_dim=args.out_dim).cuda()
        input_dim += 512


    mlp_cls = Encoder(input_dim=input_dim, out_dim=args.num_class).cuda()


    criterion          = nn.CrossEntropyLoss().cuda()
    criterion_supcon   = SupConLoss(temperature=args.temp_s).cuda()
    criterion_unsupcon = ConMeanShiftLoss(n_views=2,
                                          alpha=0.5,
                                          temperature=args.temp_u).cuda()


    params = list(mlp_cls.parameters())
    if args.use_video:
        params += (list(model.module.backbone.fast_path.layer4.parameters())
                   + list(model.module.backbone.slow_path.layer4.parameters())
                   + list(model.module.cls_head.parameters())
                   + list(v_proj.parameters()))
    if args.use_flow:
        params += (list(model_flow.module.backbone.layer4.parameters())
                   + list(model_flow.module.cls_head.parameters())
                   + list(f_proj.parameters()))
    if args.use_audio:
        params += list(audio_cls_model.parameters()) + list(a_proj.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.nepochs, eta_min=args.lr * 1e-2
    )


    modal_code = _build_modality_code(args.use_video, args.use_audio, args.use_flow)
    log_name = build_run_name(METHOD_NAME, DATASET_NAME, args.source_domain, args.target_domain,
                              modal_code, args.seed, args.appen, args.run_name)
    log_dir, model_dir, log_path = build_output_paths(
        __file__, DATASET_NAME, METHOD_NAME, args.dg_mode, log_name
    )
    print('Log path:', log_path)

    BestLoss    = float('inf')
    BestEpoch   = args.BestEpoch
    BestAcc     = args.BestAcc
    BestTestAcc = args.BestTestAcc


    if args.resumef:
        resume_file = os.path.join(model_dir, log_name + '.pt')
        print('Resuming from', resume_file)
        ckpt = torch.load(resume_file)
        starting_epoch = ckpt['epoch'] + 1
        BestLoss    = ckpt['BestLoss']
        BestEpoch   = ckpt['BestEpoch']
        BestAcc     = ckpt['BestAcc']
        BestTestAcc = ckpt['BestTestAcc']

        mlp_cls.load_state_dict(ckpt['mlp_cls_state_dict'])
        optim.load_state_dict(ckpt['optimizer'])
        if args.use_video:
            model.load_state_dict(ckpt['model_state_dict'])
            v_proj.load_state_dict(ckpt['v_proj_state_dict'])
        if args.use_flow:
            model_flow.load_state_dict(ckpt['model_flow_state_dict'])
            f_proj.load_state_dict(ckpt['f_proj_state_dict'])
        if args.use_audio:
            audio_model.load_state_dict(ckpt['audio_model_state_dict'])
            audio_cls_model.load_state_dict(ckpt['audio_cls_model_state_dict'])
            a_proj.load_state_dict(ckpt['a_proj_state_dict'])
    else:
        print('Training from scratch ...')
        starting_epoch = 0

    print('Starting epoch:', starting_epoch)


    best_state = {
        'mlp_cls_state_dict': copy.deepcopy(mlp_cls.state_dict()),
        'optimizer': copy.deepcopy(optim.state_dict()),
    }
    if args.use_video:
        best_state['model_state_dict'] = copy.deepcopy(model.state_dict())
        best_state['v_proj_state_dict'] = copy.deepcopy(v_proj.state_dict())
    if args.use_flow:
        best_state['model_flow_state_dict'] = copy.deepcopy(model_flow.state_dict())
        best_state['f_proj_state_dict'] = copy.deepcopy(f_proj.state_dict())
    if args.use_audio:
        best_state['audio_model_state_dict'] = copy.deepcopy(audio_model.state_dict())
        best_state['audio_cls_model_state_dict'] = copy.deepcopy(audio_cls_model.state_dict())
        best_state['a_proj_state_dict'] = copy.deepcopy(a_proj.state_dict())


    train_dataset = DOMAINNVIEWS(DatasetClass, **make_dataset_kwargs(args, 'train', args.source_domain, cfg, cfg_flow, source=True))
    val_dataset = DOMAINNVIEWS(DatasetClass, **make_dataset_kwargs(args, 'test', args.source_domain, cfg, cfg_flow, source=True))
    if len(args.target_domain) == 1:
        test_dataset = DOMAINNVIEWS(DatasetClass, **make_dataset_kwargs(args, 'test', args.target_domain, cfg, cfg_flow, source=False))
    else:
        test_dataset1 = DOMAINNVIEWS(DatasetClass, **make_dataset_kwargs(args, 'test', args.target_domain[0:1], cfg, cfg_flow, source=False))
        test_dataset2 = DOMAINNVIEWS(DatasetClass, **make_dataset_kwargs(args, 'test', args.target_domain[1:2], cfg, cfg_flow, source=False))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bsz,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bsz,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True, drop_last=False)
    if len(args.target_domain) == 1:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.bsz,
            num_workers=args.num_workers, shuffle=False,
            pin_memory=True, drop_last=False)
    else:
        test_loader1 = torch.utils.data.DataLoader(
            test_dataset1, batch_size=args.bsz,
            num_workers=args.num_workers, shuffle=False,
            pin_memory=True, drop_last=False)
        test_loader2 = torch.utils.data.DataLoader(
            test_dataset2, batch_size=args.bsz,
            num_workers=args.num_workers, shuffle=False,
            pin_memory=True, drop_last=False)

    dataloaders = {'train': train_loader, 'val': val_loader}


    with open(log_path, 'a') as f:
        f.write("\n\n\n")
        f.write("run_start,{}\n".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write("dg_mode,{}\n".format(args.dg_mode))
        f.write("source_domain,{}\n".format(args.source_domain))
        f.write("target_domain,{}\n".format(args.target_domain))
        f.write("lr,{}\n".format(args.lr))
        f.write("weight_decay,{}\n".format(args.weight_decay))
        f.write("bsz,{}\n".format(args.bsz))
        f.write("nepochs,{}\n".format(args.nepochs))
        f.write("seed,{}\n".format(args.seed))
        f.write("alpha,{}\n".format(args.alpha))
        f.write("beta,{}\n".format(args.beta))
        f.write("temp_s,{}\n".format(args.temp_s))
        f.write("temp_u,{}\n".format(args.temp_u))
        f.write("k,{}\n".format(args.k))
        f.write("hidden_dim,{}\n".format(args.hidden_dim))
        f.write("out_dim,{}\n".format(args.out_dim))
        modality_code = _build_modality_code(args.use_video, args.use_audio, args.use_flow) or 'none'
        f.write("modalities,{}\n".format(modality_code))
        f.write("\n")
        for epoch_i in range(starting_epoch, args.nepochs):
            print('=' * 50)
            print(f'Epoch: {epoch_i:02d}')
            print(f'  α={args.alpha:.3f}  β={args.beta:.5f}  k={args.k}')
            print('=' * 50)

            currentvalAcc = 0.0

            for split in ['train', 'val']:
                acc = count = 0
                total_loss  = 0.0
                is_train    = (split == 'train')

                mlp_cls.train(is_train)
                if args.use_video:
                    model.train(is_train);  v_proj.train(is_train)
                if args.use_flow:
                    model_flow.train(is_train); f_proj.train(is_train)
                if args.use_audio:
                    audio_cls_model.train(is_train); a_proj.train(is_train)

                print(f'\n{split.upper()} phase:')
                with tqdm.tqdm(total=len(dataloaders[split]), disable=True) as pbar:
                    for clip, flow, spec, labels in dataloaders[split]:
                        if split == 'train':
                            pred, loss = train_one_step(clip, labels, flow, spec)
                        else:
                            pred, loss = validate_one_step(clip, labels, flow, spec)

                        batch_n = pred.size(0)
                        total_loss += loss.item() * batch_n
                        _, predicted = torch.max(pred.detach().cpu(), dim=1)
                        acc   += (predicted == labels).sum().item()
                        count += batch_n

                        pbar.set_postfix_str(
                            f"Avg loss: {total_loss/count:.4f}  "
                            f"Cur loss: {loss.item():.4f}  "
                            f"Acc: {acc/count:.4f}"
                        )
                        pbar.update()

                cur_acc  = acc  / count
                cur_loss = total_loss / count

                if split == 'val':
                    currentvalAcc = cur_acc
                    if currentvalAcc >= BestAcc:
                        BestLoss  = cur_loss
                        BestEpoch = epoch_i
                        BestAcc   = cur_acc
                        best_state = {
                            'mlp_cls_state_dict': copy.deepcopy(mlp_cls.state_dict()),
                            'optimizer': copy.deepcopy(optim.state_dict()),
                        }
                        if args.use_video:
                            best_state['model_state_dict'] = copy.deepcopy(model.state_dict())
                            best_state['v_proj_state_dict'] = copy.deepcopy(v_proj.state_dict())
                        if args.use_flow:
                            best_state['model_flow_state_dict'] = copy.deepcopy(model_flow.state_dict())
                            best_state['f_proj_state_dict'] = copy.deepcopy(f_proj.state_dict())
                        if args.use_audio:
                            best_state['audio_model_state_dict'] = copy.deepcopy(audio_model.state_dict())
                            best_state['audio_cls_model_state_dict'] = copy.deepcopy(audio_cls_model.state_dict())
                            best_state['a_proj_state_dict'] = copy.deepcopy(a_proj.state_dict())

                    if args.save_checkpoint:
                        save = {
                            'epoch': epoch_i,
                            'BestLoss': BestLoss,
                            'BestEpoch': BestEpoch,
                            'BestAcc': BestAcc,
                            'BestTestAcc': BestTestAcc,
                            'optimizer': optim.state_dict(),
                            'mlp_cls_state_dict': mlp_cls.state_dict(),
                        }
                        if args.use_video:
                            save['model_state_dict']  = model.state_dict()
                            save['v_proj_state_dict'] = v_proj.state_dict()
                        if args.use_flow:
                            save['model_flow_state_dict'] = model_flow.state_dict()
                            save['f_proj_state_dict']     = f_proj.state_dict()
                        if args.use_audio:
                            save['audio_model_state_dict']     = audio_model.state_dict()
                            save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                            save['a_proj_state_dict']          = a_proj.state_dict()
                        torch.save(save, os.path.join(model_dir, log_name + '.pt'))

                f.write(f'{epoch_i},{split},{cur_loss:.6f},{cur_acc:.6f}\n')
                f.flush()

                print(f'\n{split} — Epoch {epoch_i}:  Acc={cur_acc:.4f}')
                if split == 'val':
                    print(f'  Best Val Acc:  {BestAcc:.4f}')
                    print(f'  Best Epoch:    {BestEpoch}')
                    f.write(
                        f'CurrentBest,epoch={BestEpoch},'
                        f'valAcc={BestAcc:.6f}\n'
                    )
                    f.flush()


            lr_scheduler.step()

        mlp_cls.load_state_dict(best_state['mlp_cls_state_dict'])
        if args.use_video:
            model.load_state_dict(best_state['model_state_dict'])
            v_proj.load_state_dict(best_state['v_proj_state_dict'])
        if args.use_flow:
            model_flow.load_state_dict(best_state['model_flow_state_dict'])
            f_proj.load_state_dict(best_state['f_proj_state_dict'])
        if args.use_audio:
            audio_model.load_state_dict(best_state['audio_model_state_dict'])
            audio_cls_model.load_state_dict(best_state['audio_cls_model_state_dict'])
            a_proj.load_state_dict(best_state['a_proj_state_dict'])

        mlp_cls.eval()
        if args.use_video:
            model.eval()
            v_proj.eval()
        if args.use_flow:
            model_flow.eval()
            f_proj.eval()
        if args.use_audio:
            audio_cls_model.eval()
            a_proj.eval()

        print('\nTEST phase (run once after full training):')
        if len(args.target_domain) == 1:
            test_acc = 0
            test_count = 0
            test_total_loss = 0.0
            with tqdm.tqdm(total=len(test_loader), disable=True) as pbar:
                for clip, flow, spec, labels in test_loader:
                    pred, loss = validate_one_step(clip, labels, flow, spec)
                    batch_n = pred.size(0)
                    test_total_loss += loss.item() * batch_n
                    _, predicted = torch.max(pred.detach().cpu(), dim=1)
                    test_acc += (predicted == labels).sum().item()
                    test_count += batch_n
                    pbar.update()

            BestTestAcc = test_acc / float(test_count)
            test_loss = test_total_loss / float(test_count)
            f.write(f'{BestEpoch},test,{test_loss:.6f},{BestTestAcc:.6f}\n')
        else:
            test_acc_1 = 0
            test_count_1 = 0
            test_total_loss_1 = 0.0
            with tqdm.tqdm(total=len(test_loader1), disable=True) as pbar:
                for clip, flow, spec, labels in test_loader1:
                    pred, loss = validate_one_step(clip, labels, flow, spec)
                    batch_n = pred.size(0)
                    test_total_loss_1 += loss.item() * batch_n
                    _, predicted = torch.max(pred.detach().cpu(), dim=1)
                    test_acc_1 += (predicted == labels).sum().item()
                    test_count_1 += batch_n
                    pbar.update()

            test_acc_2 = 0
            test_count_2 = 0
            test_total_loss_2 = 0.0
            with tqdm.tqdm(total=len(test_loader2), disable=True) as pbar:
                for clip, flow, spec, labels in test_loader2:
                    pred, loss = validate_one_step(clip, labels, flow, spec)
                    batch_n = pred.size(0)
                    test_total_loss_2 += loss.item() * batch_n
                    _, predicted = torch.max(pred.detach().cpu(), dim=1)
                    test_acc_2 += (predicted == labels).sum().item()
                    test_count_2 += batch_n
                    pbar.update()

            test_loss_1 = test_total_loss_1 / float(test_count_1)
            test_loss_2 = test_total_loss_2 / float(test_count_2)
            test_acc_1 = test_acc_1 / float(test_count_1)
            test_acc_2 = test_acc_2 / float(test_count_2)
            BestTestAcc = (test_acc_1 + test_acc_2) / 2.0

            f.write(f'{BestEpoch},test {args.target_domain[0]},{test_loss_1:.6f},{test_acc_1:.6f}\n')
            f.write(f'{BestEpoch},test {args.target_domain[1]},{test_loss_2:.6f},{test_acc_2:.6f}\n')
        f.flush()

        if args.save_best:
            save = {
                'epoch': BestEpoch,
                'BestLoss': BestLoss,
                'BestEpoch': BestEpoch,
                'BestAcc': BestAcc,
                'BestTestAcc': BestTestAcc,
                'optimizer': best_state['optimizer'],
                'mlp_cls_state_dict': best_state['mlp_cls_state_dict'],
            }
            if args.use_video:
                save['model_state_dict'] = best_state['model_state_dict']
                save['v_proj_state_dict'] = best_state['v_proj_state_dict']
            if args.use_flow:
                save['model_flow_state_dict'] = best_state['model_flow_state_dict']
                save['f_proj_state_dict'] = best_state['f_proj_state_dict']
            if args.use_audio:
                save['audio_model_state_dict'] = best_state['audio_model_state_dict']
                save['audio_cls_model_state_dict'] = best_state['audio_cls_model_state_dict']
                save['a_proj_state_dict'] = best_state['a_proj_state_dict']
            save_path = os.path.join(model_dir, f'{log_name}_best_{BestEpoch}.pt')
            torch.save(save, save_path)
            print(f'SavedBestModelPath: {save_path}')

        f.write(
            f'Final,BestEpoch={BestEpoch},'
            f'BestValAcc={BestAcc:.6f},BestTestAcc={BestTestAcc:.6f}\n'
        )

    print('\n' + '=' * 50)
    print('Training Complete')
    print(f'  Best Val Acc:  {BestAcc:.4f}')
    print(f'  Best Test Acc: {BestTestAcc:.4f}')
    print(f'  Best Epoch:    {BestEpoch}')
