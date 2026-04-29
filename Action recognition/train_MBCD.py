import argparse
import os
import os.path as osp
import sys

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mbcd.runtime_patches import apply_mbcd_runtime_patches

apply_mbcd_runtime_patches()

import torch.utils
import torch.utils.data
from mmaction.apis import init_recognizer
import torch
import tqdm
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.test import get_arguments
from dataloaders.dataloader_EPIC import EPICDOMAIN
from dataloaders.dataloader_HAC import HACDOMAIN
import torch.nn.functional as F
import learn2learn as l2l
import copy
from datetime import datetime


METHOD_NAME = 'MBCD'


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
    return '-'.join(str(domain) for domain in domains)


def build_run_name(method_name, dataset_name, source_domains, target_domains,
                   modal_code, seed, appen='', run_name=None):
    if run_name:
        return run_name
    appen_tag = f'_{appen}' if appen else ''
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    return (
        f'{method_name}_{dataset_name}_'
        f'{_format_domain_tag(source_domains)}_to_{_format_domain_tag(target_domains)}_'
        f'{modal_code}_seed{seed}{appen_tag}_{run_id}'
    )


def build_output_paths(script_file, dataset_name, method_name, dg_mode, run_name):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    output_root = os.path.join(script_dir, 'outputs')
    log_dir = os.path.join(output_root, 'logs', dataset_name, method_name, dg_mode)
    model_dir = os.path.join(output_root, 'models', dataset_name, method_name, dg_mode)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir, os.path.join(log_dir, run_name + '.csv')


def build_checkpoint(epoch_i, best_epoch, best_loss, best_val_acc, best_test_acc,
                     optim, mm_cls, mm_cls_ema, model_video=None,
                     model_video_ema=None, model_flow=None, model_flow_ema=None,
                     model_audio=None, model_audio_ema=None):
    save = {
        'epoch': epoch_i,
        'BestEpoch': best_epoch,
        'BestLoss': best_loss,
        'BestValAcc': best_val_acc,
        'BestTestAcc': best_test_acc,
        'optimizer': optim.state_dict(),
        'mm_cls_state_dict': mm_cls.state_dict(),
        'mm_cls_ema_state_dict': mm_cls_ema.state_dict(),
    }
    if model_video is not None:
        save['model_video_state_dict'] = model_video.state_dict()
        save['model_video_ema_state_dict'] = model_video_ema.state_dict()
    if model_flow is not None:
        save['model_flow_state_dict'] = model_flow.state_dict()
        save['model_flow_ema_state_dict'] = model_flow_ema.state_dict()
    if model_audio is not None:
        save['model_audio_state_dict'] = model_audio.state_dict()
        save['model_audio_ema_state_dict'] = model_audio_ema.state_dict()
    return save


def write_all_hparams(f, args):
    f.write('hparams_begin\n')
    for k in sorted(vars(args).keys()):
        f.write('{}={}\n'.format(k, getattr(args, k)))
    f.write('hparams_end\n')
    f.write('\n')

def train_one_step(clip, flow, spectrogram, labels, args, model_video, model_flow, model_audio, mm_cls, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion, optim):
    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()

    if args.use_video:
        new_model_video = l2l.clone_module(model_video)
    if args.use_flow:
        new_model_flow = l2l.clone_module(model_flow)
    if args.use_audio:
        new_model_audio = l2l.clone_module(model_audio)

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = new_model_video.module.backbone.get_feature(clip)  
            v_feat = (x_slow.detach(), x_fast.detach())  
        if args.use_flow:
            f_feat = new_model_flow.module.backbone.get_feature(flow)
        if args.use_audio:
            a_feat = new_model_audio.audnet.get_feature(spectrogram)

    if args.use_video:
        if args.use_dsu:
            v_feat = new_model_video.module.backbone.dsu(v_feat)
            v_feat = new_model_video.module.backbone.get_predict(v_feat)
            v_feat = new_model_video.module.backbone.dsu(v_feat)
        else:
            v_feat = new_model_video.module.backbone.get_predict(v_feat)
        v_emd = new_model_video.module.backbone.avg_pool(v_feat)
        v_emd = v_norm(v_emd)
        v_pred = v_cls(v_emd)
        v_max, _ = torch.max(torch.softmax(v_pred, dim=-1), dim=-1)
        v_score = v_max.sum().detach()
    if args.use_flow:
        if args.use_dsu:
            f_feat = new_model_flow.module.backbone.dsu(f_feat)
            f_feat = new_model_flow.module.backbone.get_predict(f_feat.detach())
            f_feat = new_model_flow.module.backbone.dsu(f_feat)
        else:
            f_feat = new_model_flow.module.backbone.get_predict(f_feat.detach())
        f_emd = new_model_flow.module.backbone.avg_pool(f_feat)
        f_emd = f_norm(f_emd)
        f_pred = f_cls(f_emd)
        f_max, _ = torch.max(torch.softmax(f_pred, dim=-1), dim=-1)
        f_score = f_max.sum().detach()
    if args.use_audio:    
        if args.use_dsu:
            a_feat = new_model_audio.audnet.dsu(a_feat)
            a_feat = new_model_audio.audnet.get_predict(a_feat.detach())
            a_feat = new_model_audio.audnet.dsu(a_feat)
        else:
            a_feat = new_model_audio.audnet.get_predict(a_feat.detach())
        a_emd = new_model_audio.audnet.avg_pool(a_feat)
        a_emd = a_norm(a_emd)
        a_pred = a_cls(a_emd)
        a_max, _ = torch.max(torch.softmax(a_pred, dim=-1), dim=-1)
        a_score = a_max.sum().detach()

    # inner
    if args.use_video:
        loss_v=criterion(v_pred, labels)
        diff_params = [p for name, p in new_model_video.module.named_parameters() if 'layer4' in name]
        grads = torch.autograd.grad(loss_v, diff_params, retain_graph=True)
        for p, grad in zip(diff_params, grads):
            p.update = - args.inner_lr * grad
        l2l.update_module(new_model_video.module)
    if args.use_flow:
        loss_f=criterion(f_pred, labels)
        diff_params = [p for name, p in new_model_flow.module.named_parameters() if 'layer4' in name]
        grads = torch.autograd.grad(loss_f, diff_params, retain_graph=True)
        for p, grad in zip(diff_params, grads):
            p.update = - args.inner_lr * grad
        l2l.update_module(new_model_flow.module)
    if args.use_audio:    
        loss_a=criterion(a_pred,labels)
        diff_params = [p for name, p in new_model_audio.audnet.named_parameters() if 'layer4' in name]
        grads = torch.autograd.grad(loss_a, diff_params, retain_graph=True)
        for p, grad in zip(diff_params, grads):
            p.update = - args.inner_lr * grad
        l2l.update_module(new_model_audio.audnet)
    
    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = new_model_video.module.backbone.get_feature(clip)  
            v_feat = (x_slow.detach(), x_fast.detach())  
        if args.use_flow:
            f_feat = new_model_flow.module.backbone.get_feature(flow)
        if args.use_audio:
            a_feat = new_model_audio.audnet.get_feature(spectrogram)

    if args.use_video:
        v_feat = new_model_video.module.backbone.get_predict(v_feat)
        v_emd = new_model_video.module.backbone.avg_pool(v_feat)
        v_emd = v_norm(v_emd)
    if args.use_flow:
        f_feat = new_model_flow.module.backbone.get_predict(f_feat.detach())
        f_emd = new_model_flow.module.backbone.avg_pool(f_feat)
        f_emd = f_norm(f_emd)
    if args.use_audio:    
        a_feat = new_model_audio.audnet.get_predict(a_feat.detach())
        a_emd = new_model_audio.audnet.avg_pool(a_feat)
        a_emd = a_norm(a_emd)

    # modality dropout
    bsz = len(labels)
    if args.use_video and args.use_flow and args.use_audio:
        v_r = (v_score / f_score + v_score / a_score) / 2
        f_r = (f_score / v_score + f_score / a_score) / 2
        a_r = (a_score / v_score + a_score / f_score) / 2
    elif args.use_video and args.use_flow:
        v_r = v_score / f_score
        f_r = f_score / v_score
    elif args.use_video and args.use_audio:
        v_r = v_score / a_score
        a_r = a_score / v_score
    elif args.use_flow and args.use_audio:
        f_r = f_score / a_score
        a_r = a_score / f_score
    
    if args.use_video and v_r > 1:
        v_p = args.modality_drop_base + (1 - args.modality_drop_base) * torch.tanh(v_r-1)
        v_mask = torch.bernoulli((1 - v_p) * torch.ones(bsz).to(labels.device)).unsqueeze(1)
        v_emd = v_emd * v_mask
    if args.use_flow and f_r > 1:
        f_p = args.modality_drop_base + (1 - args.modality_drop_base) * torch.tanh(f_r-1)
        f_mask = torch.bernoulli((1 - f_p) * torch.ones(bsz).to(labels.device)).unsqueeze(1)
        f_emd = f_emd * f_mask
    if args.use_audio and a_r > 1:
        a_p = args.modality_drop_base + (1 - args.modality_drop_base) * torch.tanh(a_r-1)
        a_mask = torch.bernoulli((1 - a_p) * torch.ones(bsz).to(labels.device)).unsqueeze(1)
        a_emd = a_emd * a_mask

    # concat-based fuse
    if args.use_video and args.use_flow and args.use_audio:
        feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
    elif args.use_video and args.use_flow:
        feat = torch.cat((v_emd, f_emd), dim=1)
    elif args.use_video and args.use_audio:
        feat = torch.cat((v_emd, a_emd), dim=1)
    elif args.use_flow and args.use_audio:
        feat = torch.cat((f_emd, a_emd), dim=1)
        
    mm_pred = mm_cls(feat)
    loss = criterion(mm_pred, labels)
    if args.use_video:
        loss = loss + loss_v
    if args.use_flow:
        loss = loss + loss_f
    if args.use_audio:
        loss = loss + loss_a

    # knowledge distillation
    if args.kl_mm_coeff != 0 or args.kl_um_coeff != 0:
        with torch.no_grad():
            if args.use_video:
                x_slow, x_fast = model_video_ema.module.backbone.get_feature(clip)  
                v_feat = (x_slow.detach(), x_fast.detach())  
                v_feat = model_video_ema.module.backbone.get_predict(v_feat)
                v_emd = model_video_ema.module.backbone.avg_pool(v_feat)
                v_emd = v_norm(v_emd)
            if args.use_flow:
                f_feat = model_flow_ema.module.backbone.get_feature(flow)
                f_feat = model_flow_ema.module.backbone.get_predict(f_feat.detach())
                f_emd = model_flow_ema.module.backbone.avg_pool(f_feat)
                f_emd = f_norm(f_emd)
            if args.use_audio:
                a_feat = model_audio_ema.audnet.get_feature(spectrogram)
                a_feat = model_audio_ema.audnet.get_predict(a_feat.detach())
                a_emd = model_audio_ema.audnet.avg_pool(a_feat)
                a_emd = a_norm(a_emd)

            if args.use_video and args.use_flow and args.use_audio:
                feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
            elif args.use_video and args.use_flow:
                feat = torch.cat((v_emd, f_emd), dim=1)
            elif args.use_video and args.use_audio:
                feat = torch.cat((v_emd, a_emd), dim=1)
            elif args.use_flow and args.use_audio:
                feat = torch.cat((f_emd, a_emd), dim=1)

        mm_pred_ema = mm_cls_ema(feat).detach()
        mm_prob_ema = F.softmax(mm_pred_ema, dim=-1).clamp_min(1e-12)

        if args.kl_mm_coeff != 0:
            mm_log_prob = F.log_softmax(mm_pred, dim=-1)
            kd_loss_mm = F.kl_div(mm_log_prob, mm_prob_ema, reduction='batchmean')
            loss = loss + kd_loss_mm * args.kl_mm_coeff

        if args.kl_um_coeff != 0:
            if args.use_video:
                v_log_prob = F.log_softmax(v_pred, dim=-1)
                kd_loss_v = F.kl_div(v_log_prob, mm_prob_ema, reduction='batchmean')
                loss = loss + kd_loss_v * args.kl_um_coeff
            if args.use_flow:
                f_log_prob = F.log_softmax(f_pred, dim=-1)
                kd_loss_f = F.kl_div(f_log_prob, mm_prob_ema, reduction='batchmean')
                loss = loss + kd_loss_f * args.kl_um_coeff
            if args.use_audio:
                a_log_prob = F.log_softmax(a_pred, dim=-1)
                kd_loss_a = F.kl_div(a_log_prob, mm_prob_ema, reduction='batchmean')
                loss = loss + kd_loss_a * args.kl_um_coeff

    optim.zero_grad()
    loss.backward()
    optim.step()

    return mm_pred, loss

def validate_one_step(clip, flow, spectrogram, labels, args, model_video, model_flow, model_audio, mm_cls, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion):
    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model_video.module.backbone.get_feature(clip) 
            v_feat = (x_slow, x_fast)
            v_feat = model_video.module.backbone.get_predict(v_feat)
            v_emd = model_video.module.backbone.avg_pool(v_feat)
            v_emd = v_norm(v_emd)
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)  
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_emd = model_flow.module.backbone.avg_pool(f_feat)
            f_emd = f_norm(f_emd)
        if args.use_audio:
            a_feat = model_audio.audnet.get_feature(spectrogram)
            a_feat = model_audio.audnet.get_predict(a_feat)
            a_emd = model_audio.audnet.avg_pool(a_feat)
            a_emd = a_norm(a_emd)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, a_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, a_emd), dim=1)

        predict = mm_cls(feat)
        loss = criterion(predict, labels)

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model_video_ema.module.backbone.get_feature(clip) 
            v_feat = (x_slow, x_fast)
            v_feat = model_video_ema.module.backbone.get_predict(v_feat)
            v_emd = model_video_ema.module.backbone.avg_pool(v_feat)
            v_emd = v_norm(v_emd)
        if args.use_flow:
            f_feat = model_flow_ema.module.backbone.get_feature(flow)  
            f_feat = model_flow_ema.module.backbone.get_predict(f_feat)
            f_emd = model_flow_ema.module.backbone.avg_pool(f_feat)
            f_emd = f_norm(f_emd)
        if args.use_audio:
            a_feat = model_audio_ema.audnet.get_feature(spectrogram)
            a_feat = model_audio_ema.audnet.get_predict(a_feat)
            a_emd = model_audio_ema.audnet.avg_pool(a_feat)
            a_emd = a_norm(a_emd)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, a_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, a_emd), dim=1)

        predict_ema = mm_cls_ema(feat)
        loss_ema = criterion(predict_ema, labels)
    
    return predict, loss, predict_ema, loss_ema

def test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion):
    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model_video_ema.module.backbone.get_feature(clip) 
            v_feat = (x_slow, x_fast)
            v_feat = model_video_ema.module.backbone.get_predict(v_feat)
            v_emd = model_video_ema.module.backbone.avg_pool(v_feat)
            v_emd = v_norm(v_emd)
        if args.use_flow:
            f_feat = model_flow_ema.module.backbone.get_feature(flow)  
            f_feat = model_flow_ema.module.backbone.get_predict(f_feat)
            f_emd = model_flow_ema.module.backbone.avg_pool(f_feat)
            f_emd = f_norm(f_emd)
        if args.use_audio:
            a_feat = model_audio_ema.audnet.get_feature(spectrogram)
            a_feat = model_audio_ema.audnet.get_predict(a_feat)
            a_emd = model_audio_ema.audnet.avg_pool(a_feat)
            a_emd = a_norm(a_emd)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, a_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, a_emd), dim=1)

        predict_ema = mm_cls_ema(feat)
        loss_ema = criterion(predict_ema, labels)
    
    return predict_ema, loss_ema

class PredHead(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(PredHead, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape)

    def forward(self, feat):
        feat = self.layernorm(feat)
        return feat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument('--use_dsu', default=True, action='store_true', help="Enable DSU: UNCERTAINTY MODELING FOR OUT-OF-DISTRIBUTION GENERALIZATION")
    parser.add_argument('--inner_lr', type=float, default=1e-4)
    parser.add_argument('--ema_beta', type=float, default=0.999)
    parser.add_argument('--kl_mm_coeff', type=float, default=1.0)
    parser.add_argument('--kl_um_coeff', type=float, default=1.0)
    parser.add_argument('--modality_drop_base', type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--appen', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    
    # Modality Settings
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow', action='store_true')

    # Dataset & Path Settings
    parser.add_argument('--dataset', type=str, required=True, choices=['epic', 'hac'])
    parser.add_argument('-s', '--source_domain', nargs='+', required=True, help='<Required> Set source_domain')
    parser.add_argument('-t', '--target_domain', nargs='+', required=True, help='<Required> Set target_domain')

    parser.add_argument('--datapath', type=str, default='/path/to/DATA_ROOT',
                        help='datapath')
    parser.add_argument("--num_class", type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=4)

    # Logging & Results
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestValAcc', type=float, default=0)
    parser.add_argument('--BestTestAcc', type=float, default=0)

    # System Settings
    parser.add_argument('--gpu', type=int, default=1)
    
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

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load config files and checkpoint
    config_file = osp.join(SCRIPT_DIR, 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py')
    checkpoint_file = osp.join(SCRIPT_DIR, 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth')
    config_file_flow = osp.join(SCRIPT_DIR, 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py')
    checkpoint_file_flow = osp.join(SCRIPT_DIR, 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth')

    # assign the desired device.
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    batch_size = args.bsz
    input_dim = 0
    model_video, model_flow, model_audio = None, None, None
    cfg_video, cfg_flow = None, None

    if args.use_video:
        model_video = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        cfg_video = model_video.cfg
        model_video = torch.nn.DataParallel(model_video)

        v_norm = LayerNorm(2304).cuda()
        v_cls = PredHead(input_dim=2304, out_dim=args.num_class).cuda()

        input_dim += 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        f_norm = LayerNorm(2048).cuda()
        f_cls = PredHead(input_dim=2048, out_dim=args.num_class).cuda()

        input_dim += 2048

    if args.use_audio:
        audio_args = get_arguments()
        model_audio = AVENet(audio_args)
        checkpoint = torch.load(osp.join(SCRIPT_DIR, "pretrained_models/vggsound_avgpool.pth.tar"), map_location=device)
        model_audio.load_state_dict(checkpoint['model_state_dict'])
        model_audio = model_audio.cuda()

        a_norm = LayerNorm(512).cuda()
        a_cls = PredHead(input_dim=512, out_dim=args.num_class).cuda()

        input_dim += 512

    mm_cls = PredHead(input_dim=input_dim, out_dim=args.num_class).cuda()

    modality_code = _build_modality_code(args.use_video, args.use_audio, args.use_flow)
    log_name = build_run_name(METHOD_NAME, DATASET_NAME, args.source_domain, args.target_domain,
                              modality_code, args.seed, args.appen, args.run_name)
    log_dir, model_dir, log_path = build_output_paths(
        __file__, DATASET_NAME, METHOD_NAME, args.dg_mode, log_name
    )
    print('Log path:', log_path)

    criterion = nn.CrossEntropyLoss()

    params = list(mm_cls.parameters())
    if args.use_video:
        params = params + list(model_video.module.backbone.fast_path.layer4.parameters()) + list(model_video.module.backbone.slow_path.layer4.parameters()) + list(v_cls.parameters())
    if args.use_flow:
        params = params + list(model_flow.module.backbone.layer4.parameters()) + list(f_cls.parameters())
    if args.use_audio:
        params = params + list(model_audio.audnet.layer4.parameters()) + list(a_cls.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
   
    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestValAcc = args.BestValAcc
    BestTestAcc = args.BestTestAcc

    train_dataset = DatasetClass(**make_dataset_kwargs(args, 'train', args.source_domain, cfg_video, cfg_flow, source=True))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    validate_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.source_domain, cfg_video, cfg_flow, source=True))
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    if len(args.target_domain) == 1:
        test_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain, cfg_video, cfg_flow, source=False))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)                                            
    else:
        test_dataset1 = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain[0:1], cfg_video, cfg_flow, source=False))
        test_dataset2 = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain[1:], cfg_video, cfg_flow, source=False))
        test_dataloader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)                                            
        test_dataloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)   
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader}
    # open log: always append and record hyperparameters for each run
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("\n\n\n\n")
        f.write("run_start,{}\n".format(datetime.now().isoformat()))
        f.write("dg_mode,{}\n".format(args.dg_mode))
        write_all_hparams(f, args)
        # ema model
        if args.use_video:
            model_video_ema = copy.deepcopy(model_video)
            model_video_ema.eval()
        else:
            model_video_ema = None
        if args.use_flow:
            model_flow_ema = copy.deepcopy(model_flow)
            model_flow_ema.eval()
        else:
            model_flow_ema = None
        if args.use_audio:
            model_audio_ema = copy.deepcopy(model_audio)
            model_audio_ema.eval()
        else:
            model_audio_ema = None
        mm_cls_ema = copy.deepcopy(mm_cls)
        mm_cls_ema.eval()
        # iteration
        for epoch_i in range(1, args.nepochs+1):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0
                acc_ema = 0
                print(split)
                mm_cls.train(split == 'train')
                if args.use_video:
                    model_video.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                if args.use_audio:
                    model_audio.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split]), disable=True) as pbar:
                    for (clip, flow, spectrogram, labels) in dataloaders[split]:
                        if split=='train':
                            predict1, loss = train_one_step(clip, flow, spectrogram, labels, args, model_video, model_flow, model_audio, mm_cls, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion, optim)
                            new_video_dict = {}
                            new_flow_dict = {}
                            new_audio_dict = {}
                            new_cls_dict = {}
                            # exponential moving average
                            if args.use_video:
                                for (name, param_q), (_, param_k) in zip(model_video.state_dict().items(), model_video_ema.state_dict().items()):
                                    new_video_dict[name] = param_k.data.detach().clone() * args.ema_beta + param_q.data.detach().clone() * (1 - args.ema_beta)
                                model_video_ema.load_state_dict(new_video_dict)
                            if args.use_flow:
                                for (name, param_q), (_, param_k) in zip(model_flow.state_dict().items(), model_flow_ema.state_dict().items()):
                                    new_flow_dict[name] = param_k.data.detach().clone() * args.ema_beta + param_q.data.detach().clone() * (1 - args.ema_beta)
                                model_flow_ema.load_state_dict(new_flow_dict)
                            if args.use_audio:
                                for (name, param_q), (_, param_k) in zip(model_audio.state_dict().items(), model_audio_ema.state_dict().items()):
                                    new_audio_dict[name] = param_k.data.detach().clone() * args.ema_beta + param_q.data.detach().clone() * (1 - args.ema_beta)
                                model_audio_ema.load_state_dict(new_audio_dict)
                            for (name, param_q), (_, param_k) in zip(mm_cls.state_dict().items(), mm_cls_ema.state_dict().items()):
                                new_cls_dict[name] = param_k.data.detach().clone() * args.ema_beta + param_q.data.detach().clone() * (1 - args.ema_beta)                            
                            mm_cls_ema.load_state_dict(new_cls_dict)
                        else:
                            predict1, loss, predict1_ema, loss_ema = validate_one_step(clip, flow, spectrogram, labels, args, model_video, model_flow, model_audio, mm_cls, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)
                            _, predict_ema = torch.max(predict1_ema.detach().cpu(), dim=1)
                            acc1_ema = (predict_ema == labels).sum().item()
                            acc_ema += int(acc1_ema)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)
                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str("Average loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count), acc / float(count)))
                        pbar.update()

                    cur_loss = total_loss / float(count)
                    cur_acc = acc / float(count)
                    if split == 'val':
                        cur_acc = acc_ema / float(count)
                        if cur_acc >= BestValAcc:
                            BestLoss = cur_loss
                            BestEpoch = epoch_i
                            BestValAcc = cur_acc
                            if args.save_best:
                                save = build_checkpoint(
                                    epoch_i, BestEpoch, BestLoss, BestValAcc, BestTestAcc,
                                    optim, mm_cls, mm_cls_ema,
                                    model_video if args.use_video else None,
                                    model_video_ema if args.use_video else None,
                                    model_flow if args.use_flow else None,
                                    model_flow_ema if args.use_flow else None,
                                    model_audio if args.use_audio else None,
                                    model_audio_ema if args.use_audio else None,
                                )
                                torch.save(save, os.path.join(model_dir, log_name + '_best_{}.pt'.format(epoch_i)))

                    if split == 'val' and args.save_checkpoint:
                        save = build_checkpoint(
                            epoch_i, BestEpoch, BestLoss, BestValAcc, BestTestAcc,
                            optim, mm_cls, mm_cls_ema,
                            model_video if args.use_video else None,
                            model_video_ema if args.use_video else None,
                            model_flow if args.use_flow else None,
                            model_flow_ema if args.use_flow else None,
                            model_audio if args.use_audio else None,
                            model_audio_ema if args.use_audio else None,
                        )
                        torch.save(save, os.path.join(model_dir, log_name + '.pt'))

                    f.write("{},{},{},{}\n".format(epoch_i, split, cur_loss, cur_acc))
                    if split == 'val':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{} \n".format(BestEpoch, BestLoss, BestValAcc))
                    f.flush()

            print('BestEpoch ', BestEpoch)
            print('BestValAcc ', BestValAcc)

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{} \n".format(BestEpoch, BestLoss, BestValAcc))
        f.flush()

        # test
        print('test')
        if len(args.target_domain) == 1:
            acc = 0
            count = 0
            total_loss = 0
            with tqdm.tqdm(total=len(test_dataloader), disable=True) as pbar:
                for (i, (clip, flow, spectrogram, labels)) in enumerate(test_dataloader):
                    predict1, loss = test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)

                    total_loss += loss.item() * batch_size
                    _, predict = torch.max(predict1.detach().cpu(), dim=1)

                    acc1 = (predict == labels).sum().item()
                    acc += int(acc1)
                    count += predict1.size()[0]
                    pbar.set_postfix_str(
                        "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                loss.item(),
                                                                                                acc / float(count)))
                    pbar.update()

                BestTestAcc = acc / float(count)
                print('TestAcc ', BestTestAcc)
                f.write("{},test,{},{}\n".format(BestEpoch, total_loss / float(count), BestTestAcc))
                f.flush()
        else:
            acc = 0
            count = 0
            total_loss = 0
            with tqdm.tqdm(total=len(test_dataloader1), disable=True) as pbar:
                for (i, (clip, flow, spectrogram, labels)) in enumerate(test_dataloader1):
                    predict1, loss = test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)

                    total_loss += loss.item() * batch_size
                    _, predict = torch.max(predict1.detach().cpu(), dim=1)

                    acc1 = (predict == labels).sum().item()
                    acc += int(acc1)
                    count += predict1.size()[0]
                    pbar.set_postfix_str(
                        "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                loss.item(),
                                                                                                acc / float(count)))
                    pbar.update()

                test_loss_1 = total_loss / float(count)
                test_acc_1 = acc / float(count)
                print(f'Test {args.target_domain[0]} ', test_acc_1)
                f.write("{},test {},{},{}\n".format(BestEpoch, args.target_domain[0], test_loss_1, test_acc_1))
                f.flush()
            acc = 0
            count = 0
            total_loss = 0
            with tqdm.tqdm(total=len(test_dataloader2), disable=True) as pbar:
                for (i, (clip, flow, spectrogram, labels)) in enumerate(test_dataloader2):
                    predict1, loss = test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)

                    total_loss += loss.item() * batch_size
                    _, predict = torch.max(predict1.detach().cpu(), dim=1)

                    acc1 = (predict == labels).sum().item()
                    acc += int(acc1)
                    count += predict1.size()[0]
                    pbar.set_postfix_str(
                        "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                loss.item(),
                                                                                                acc / float(count)))
                    pbar.update()

                test_loss_2 = total_loss / float(count)
                test_acc_2 = acc / float(count)
                BestTestAcc = (test_acc_1 + test_acc_2) / 2.0
                print(f'Test {args.target_domain[1]} ', test_acc_2)
                f.write("{},test {},{},{}\n".format(BestEpoch, args.target_domain[1], test_loss_2, test_acc_2))
                f.flush()

        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(
            BestEpoch, BestLoss, BestValAcc, BestTestAcc))
        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(
            BestEpoch, BestLoss, BestValAcc, BestTestAcc))
        f.flush()

    f.close()
