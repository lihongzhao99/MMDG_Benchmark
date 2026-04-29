import copy
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datetime import datetime
from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloaders.dataloader_EPIC import EPICDOMAIN
from dataloaders.dataloader_HAC import HACDOMAIN
import torch.nn.functional as F
import torch.autograd as autograd
from typing import List



METHOD_NAME = 'CMRF'


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


def write_all_hparams(f, args):
    f.write('hparams_begin\n')
    for k in sorted(vars(args).keys()):
        f.write('{}={}\n'.format(k, getattr(args, k)))
    f.write('hparams_end\n')
    f.write('\n')



class SupConLoss(nn.Module):
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


def norm(tensor_list: List[torch.tensor], p=2):
    """Compute p-norm for tensor list"""
    return torch.cat([x.flatten() for x in tensor_list]).norm(p)


def mix_feature(m1_f, m2_f, alpha1=None, alpha2=None):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    bsz = m1_f.shape[0]
    lam1 = np.zeros(bsz)
    lam2 = np.zeros(bsz)
    if alpha1 > 0:
        i = 0
        while i < bsz:
            lam = np.random.beta(alpha1, alpha1)
            if lam > 0.5:
                lam1[i] = lam
                i += 1
            else:
                continue
        lam1 = torch.tensor(lam1).cuda().float().unsqueeze(-1)
    else:
        lam1 = 1.
    if alpha2 > 0:
        i = 0
        while i < bsz:
            lam = np.random.beta(alpha2, alpha2)
            if lam < 0.5:
                lam2[i] = lam
                i += 1
            else:
                continue
        lam2 = torch.tensor(lam2).cuda().float().unsqueeze(-1)
    else:
        lam2 = 0.

    f1 = copy.deepcopy(m1_f)
    f2 = copy.deepcopy(m2_f)
    for i in range(lam1.shape[0]):
        f1[i] = lam1[i] * m1_f[i] + (1. - lam1[i]) * m2_f[i]
        f2[i] = lam2[i] * m1_f[i] + (1. - lam2[i]) * m2_f[i]

    return f1, f2, [lam1, lam2]


def train_one_step(clip, labels, flow, spectrogram, global_step):
    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = (x_slow.detach(), x_fast.detach())
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)

    if args.use_video:
        v_feat_ = model.module.backbone.get_predict(v_feat)
        predict1, v_emd = model.module.cls_head(v_feat_)
        v_dim = int(v_emd.shape[1] / 2)
        # print(v_emd.shape, v_feat[0].shape, v_feat[1].shape)

    if args.use_flow:
        f_feat_ = model_flow.module.backbone.get_predict(f_feat.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat_)
        f_dim = int(f_emd.shape[1] / 2)

    if args.use_audio:
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
        a_dim = int(audio_emd.shape[1] / 2)

    loss = 0
    if args.CM_mixup and args.SMA:
        if args.use_video and args.use_flow and args.use_audio:

            with torch.no_grad():
                x_slow_sma, x_fast_sma = model_sma.module.backbone.get_feature(clip)
                v_feat_sma = (x_slow_sma.detach(), x_fast_sma.detach())
                v_feat_sma_ = model_sma.module.backbone.get_predict(v_feat_sma)
                _, v_emd_sma = model_sma.module.cls_head(v_feat_sma_)
                f_feat_sma = model_flow_sma.module.backbone.get_feature(flow)
                f_feat_sma_ = model_flow_sma.module.backbone.get_predict(f_feat_sma.detach())
                _, f_emd_sma = model_flow_sma.module.cls_head(f_feat_sma_)
                _, audio_emd_sma = audio_cls_model_sma(audio_feat.detach())

                v_mix_proj_sma = v_proj_m_sma(v_emd_sma)
                f_mix_proj_sma = f_proj_m_sma(f_emd_sma)
                a_mix_proj_sma = a_proj_m_sma(audio_emd_sma)

                v_emd_mix_sma1, f_emd_mix_sma1, _ = mix_feature(v_mix_proj_sma, f_mix_proj_sma, args.mix_alpha, args.mix_alpha)
                v_emd_mix_sma2, audio_emd_mix_sma1, _ = mix_feature(v_mix_proj_sma, a_mix_proj_sma, args.mix_alpha, args.mix_alpha)
                f_emd_mix_sma2, audio_emd_mix_sma2, _ = mix_feature(f_mix_proj_sma, a_mix_proj_sma, args.mix_alpha, args.mix_alpha)

            v_mix_proj = v_proj_m(v_emd)
            f_mix_proj = f_proj_m(f_emd)
            a_mix_proj = a_proj_m(audio_emd)

            if args.contrast:
                emd_proj = torch.stack([v_mix_proj, f_mix_proj, a_mix_proj], dim=1)

                # sup contrast
                loss_mix_contrast = criterion_contrast(emd_proj, labels)
                loss += args.alpha_contrast * loss_mix_contrast

            if args.distill and global_step > args.sma_start_step:  #
                v_emd_tea1 = v_emd_mix_sma1 / torch.norm(v_emd_mix_sma1, dim=1, keepdim=True)
                v_emd_tea2 = v_emd_mix_sma2 / torch.norm(v_emd_mix_sma2, dim=1, keepdim=True)
                v_emd_stu = v_mix_proj / torch.norm(v_mix_proj, dim=1, keepdim=True)
                f_emd_tea1 = f_emd_mix_sma1 / torch.norm(f_emd_mix_sma1, dim=1, keepdim=True)
                f_emd_tea2 = f_emd_mix_sma2 / torch.norm(f_emd_mix_sma2, dim=1, keepdim=True)
                f_emd_stu = f_mix_proj / torch.norm(f_mix_proj, dim=1, keepdim=True)
                a_emd_tea1 = audio_emd_mix_sma1 / torch.norm(audio_emd_mix_sma1, dim=1, keepdim=True)
                a_emd_tea2 = audio_emd_mix_sma2 / torch.norm(audio_emd_mix_sma2, dim=1, keepdim=True)
                a_emd_stu = a_mix_proj / torch.norm(a_mix_proj, dim=1, keepdim=True)
                loss += args.distill_coef * (torch.mean(torch.norm(v_emd_tea1.detach() - v_emd_stu, dim=1)) +
                                             torch.mean(torch.norm(v_emd_tea2.detach() - v_emd_stu, dim=1)) +
                                             torch.mean(torch.norm(f_emd_tea1.detach() - f_emd_stu, dim=1)) +
                                             torch.mean(torch.norm(f_emd_tea2.detach() - f_emd_stu, dim=1)) +
                                             1/2*torch.mean(torch.norm(a_emd_tea1.detach() - a_emd_stu, dim=1)) +
                                             1 / 2 * torch.mean(torch.norm(a_emd_tea2.detach() - a_emd_stu, dim=1))
                                            )

            loss += args.mix_coef * (criterion(predict1, labels) + criterion(f_predict, labels) + 1./2 * criterion(audio_predict, labels))

        elif args.use_video and args.use_audio:
            with torch.no_grad():
                x_slow_sma, x_fast_sma = model_sma.module.backbone.get_feature(clip)
                v_feat_sma = (x_slow_sma.detach(), x_fast_sma.detach())
                v_feat_sma_ = model_sma.module.backbone.get_predict(v_feat_sma)
                _, v_emd_sma = model_sma.module.cls_head(v_feat_sma_)
                _, audio_emd_sma = audio_cls_model_sma(audio_feat.detach())

                v_mix_proj_sma = v_proj_m_sma(v_emd_sma)
                a_mix_proj_sma = a_proj_m_sma(audio_emd_sma)

                v_emd_mix_sma, audio_emd_mix_sma, _ = mix_feature(v_mix_proj_sma, a_mix_proj_sma, args.mix_alpha, args.mix_alpha)  # teacher features

                # v_emd_mix_sma = v_mix_proj_sma
                # audio_emd_mix_sma = a_mix_proj_sma

            v_mix_proj = v_proj_m(v_emd)
            a_mix_proj = a_proj_m(audio_emd)

            if args.contrast:
                emd_proj = torch.stack([v_mix_proj, a_mix_proj], dim=1)

                # sup contrast
                loss_mix_contrast = criterion_contrast(emd_proj, labels)
                loss += args.alpha_contrast * loss_mix_contrast

            if args.distill and global_step > args.sma_start_step:  # 現在是SMA的模型向online模型蒸餾，能否將online模型變得比SMA模型更好，如果不能能否改為SMA的模型自蒸餾
                v_emd_tea = v_emd_mix_sma / torch.norm(v_emd_mix_sma, dim=1, keepdim=True)
                v_emd_stu = v_mix_proj / torch.norm(v_mix_proj, dim=1, keepdim=True)
                a_emd_tea = audio_emd_mix_sma / torch.norm(audio_emd_mix_sma, dim=1, keepdim=True)
                a_emd_stu = a_mix_proj / torch.norm(a_mix_proj, dim=1, keepdim=True)
                loss += args.distill_coef * (torch.mean(torch.norm(v_emd_tea.detach() - v_emd_stu, dim=1)) + 1/2 *torch.mean(
                    torch.norm(a_emd_tea.detach() - a_emd_stu, dim=1)))


            loss += args.mix_coef * (criterion(predict1, labels) + 1./2 * criterion(audio_predict, labels))

        elif args.use_video and args.use_flow:
            with torch.no_grad():
                x_slow_sma, x_fast_sma = model_sma.module.backbone.get_feature(clip)
                v_feat_sma = (x_slow_sma.detach(), x_fast_sma.detach())
                v_feat_sma_ = model_sma.module.backbone.get_predict(v_feat_sma)
                _, v_emd_sma = model_sma.module.cls_head(v_feat_sma_)
                f_feat_sma = model_flow_sma.module.backbone.get_feature(flow)
                f_feat_sma_ = model_flow_sma.module.backbone.get_predict(f_feat_sma.detach())
                _, f_emd_sma = model_flow_sma.module.cls_head(f_feat_sma_)

                v_mix_proj_sma = v_proj_m_sma(v_emd_sma)
                f_mix_proj_sma = f_proj_m_sma(f_emd_sma)

                v_emd_mix_sma, f_emd_mix_sma, _ = mix_feature(v_mix_proj_sma, f_mix_proj_sma, args.mix_alpha,
                                                                  args.mix_alpha)  # teacher features

            v_mix_proj = v_proj_m(v_emd)
            f_mix_proj = f_proj_m(f_emd)

            if args.contrast:
                emd_proj = torch.stack([v_mix_proj, f_mix_proj], dim=1)

                # sup contrast
                loss_mix_contrast = criterion_contrast(emd_proj, labels)
                loss += args.alpha_contrast * loss_mix_contrast

            if args.distill and global_step > args.sma_start_step:  # 現在是SMA的模型向online模型蒸餾，能否將online模型變得比SMA模型更好，如果不能能否改為SMA的模型自蒸餾
                v_emd_tea = v_emd_mix_sma / torch.norm(v_emd_mix_sma, dim=1, keepdim=True)
                v_emd_stu = v_mix_proj / torch.norm(v_mix_proj, dim=1, keepdim=True)
                f_emd_tea = f_emd_mix_sma / torch.norm(f_emd_mix_sma, dim=1, keepdim=True)
                f_emd_stu = f_mix_proj / torch.norm(f_mix_proj, dim=1, keepdim=True)
                loss += args.distill_coef * (
                            torch.mean(torch.norm(v_emd_tea.detach() - v_emd_stu, dim=1)) + torch.mean(
                        torch.norm(f_emd_tea.detach() - f_emd_stu, dim=1)))

            loss += args.mix_coef * (criterion(predict1, labels) + criterion(f_predict, labels))

        elif args.use_flow and args.use_audio:
            with torch.no_grad():
                f_feat_sma = model_flow_sma.module.backbone.get_feature(flow)
                f_feat_sma_ = model_flow_sma.module.backbone.get_predict(f_feat_sma.detach())
                _, f_emd_sma = model_flow_sma.module.cls_head(f_feat_sma_)
                _, audio_emd_sma = audio_cls_model_sma(audio_feat.detach())

                f_mix_proj_sma = f_proj_m_sma(f_emd_sma)
                a_mix_proj_sma = a_proj_m_sma(audio_emd_sma)

                f_emd_mix_sma, audio_emd_mix_sma, _ = mix_feature(f_mix_proj_sma, a_mix_proj_sma, args.mix_alpha,
                                                                  args.mix_alpha)  # teacher features

            f_mix_proj = f_proj_m(f_emd)
            a_mix_proj = a_proj_m(audio_emd)

            if args.contrast:
                emd_proj = torch.stack([f_mix_proj, a_mix_proj], dim=1)

                # sup contrast
                loss_mix_contrast = criterion_contrast(emd_proj, labels)
                loss += args.alpha_contrast * loss_mix_contrast

            if args.distill and global_step > args.sma_start_step:  # 現在是SMA的模型向online模型蒸餾，能否將online模型變得比SMA模型更好，如果不能能否改為SMA的模型自蒸餾
                f_emd_tea = f_emd_mix_sma / torch.norm(f_emd_mix_sma, dim=1, keepdim=True)
                f_emd_stu = f_mix_proj / torch.norm(f_mix_proj, dim=1, keepdim=True)
                a_emd_tea = audio_emd_mix_sma / torch.norm(audio_emd_mix_sma, dim=1, keepdim=True)
                a_emd_stu = a_mix_proj / torch.norm(a_mix_proj, dim=1, keepdim=True)
                loss += args.distill_coef * (
                            torch.mean(torch.norm(f_emd_tea.detach() - f_emd_stu, dim=1)) + 1. / 2 * torch.mean(
                        torch.norm(a_emd_tea.detach() - a_emd_stu, dim=1)))

            loss += args.mix_coef * (criterion(f_predict, labels) + 1. / 2 * criterion(audio_predict, labels))

    if args.use_video and args.use_flow and args.use_audio:
        feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
    elif args.use_video and args.use_flow:
        feat = torch.cat((v_emd, f_emd), dim=1)
    elif args.use_video and args.use_audio:
        feat = torch.cat((v_emd, audio_emd), dim=1)
    elif args.use_flow and args.use_audio:
        feat = torch.cat((f_emd, audio_emd), dim=1)
    elif args.use_video and not args.use_flow and not args.use_audio:
        feat = v_emd
    elif not args.use_video and args.use_flow and not args.use_audio:
        feat = f_emd
    elif not args.use_video and not args.use_flow and args.use_audio:
        feat = audio_emd

    predict = mlp_cls(feat)
    loss += criterion(predict, labels)

    if args.vanilla_learning:
        if args.DG_algorithm == 'SAM':
            dummy_random_state = np.random.RandomState(0)
            rho = (0.05, dummy_random_state.choice([0.01, 0.02, 0.05, 0.1]))  # 0.2, 0.5
            rho = rho[0]

            optim.zero_grad()

            # 1. eps(w) = rho * g(w) / g(w).norm(2)
            #           = (rho / g(w).norm(2)) * g(w)
            loss.backward()
            intermediate_grads = list()
            for _, p in mlp_cls.named_parameters():
                intermediate_grads.append(p.grad)

            if args.use_video:
                for _, p in model.module.backbone.fast_path.layer4.named_parameters():
                    # intermediate_grads += list(p.grad)
                    intermediate_grads.append(p.grad)
                for _, p in model.module.backbone.slow_path.layer4.named_parameters():
                    # intermediate_grads += p.grad
                    intermediate_grads.append(p.grad)
                for name, p in model.module.cls_head.named_parameters():
                    if p.grad is None:
                        pass
                    else:
                        # intermediate_grads += p.grad
                        intermediate_grads.append(p.grad)
            if args.use_flow:
                for _, p in model_flow.module.backbone.layer4.named_parameters():
                    # intermediate_grads += p.grad
                    intermediate_grads.append(p.grad)
                for name, p in model_flow.module.cls_head.named_parameters():
                    if p.grad is None:
                        pass
                    else:
                        intermediate_grads.append(p.grad)
                    # intermediate_grads += p.grad
            if args.use_audio:
                for name, p in audio_cls_model.named_parameters():
                    if p.grad is None:
                        pass
                    else:
                        intermediate_grads.append(p.grad)
                    # if str(name).split('.')[0] != 'fc':
                    #     intermediate_grads += p.grad

            scale = rho / norm(intermediate_grads)
            eps = [g * scale for g in intermediate_grads]
            eps_ = copy.deepcopy(eps)

            # 2. w' = w + eps(w)
            index = 0
            with torch.no_grad():
                for _, p in mlp_cls.named_parameters():
                    p.add_(eps[index])
                    index += 1
                eps = eps[index:]
                index = 0

                if args.use_video:
                    for _, p in model.module.backbone.fast_path.layer4.named_parameters():
                        p.add_(eps[index])
                        index += 1
                    for _, p in model.module.backbone.slow_path.layer4.named_parameters():
                        p.add_(eps[index])
                        index += 1
                    for name, p in model.module.cls_head.named_parameters():
                        if p.grad is None:
                            pass
                        else:
                            p.add_(eps[index])
                            index += 1
                    eps = eps[index:]
                    index = 0
                if args.use_flow:
                    for _, p in model_flow.module.backbone.layer4.named_parameters():
                        p.add_(eps[index])
                        index += 1
                    for name, p in model_flow.module.cls_head.named_parameters():
                        if p.grad is None:
                            pass
                        else:
                            p.add_(eps[index])
                            index += 1
                    eps = eps[index:]
                    index = 0
                if args.use_audio:
                    for name, p in audio_cls_model.named_parameters():
                        if p.grad is None:
                            pass
                        else:
                            p.add_(eps[index])
                            index += 1

            # 3. w = w - lr * g(w')
            if args.use_video:
                v_feat_ = model.module.backbone.get_predict(v_feat)
                predict1, v_emd = model.module.cls_head(v_feat_)

            if args.use_flow:
                f_feat_ = model_flow.module.backbone.get_predict(f_feat.detach())
                f_predict, f_emd = model_flow.module.cls_head(f_feat_)

            if args.use_audio:
                audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

            if args.use_video and args.use_flow and args.use_audio:
                feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
            elif args.use_video and args.use_flow:
                feat = torch.cat((v_emd, f_emd), dim=1)
            elif args.use_video and args.use_audio:
                feat = torch.cat((v_emd, audio_emd), dim=1)
            elif args.use_flow and args.use_audio:
                feat = torch.cat((f_emd, audio_emd), dim=1)
            elif args.use_video and not args.use_flow and not args.use_audio:
                feat = v_emd
            elif not args.use_video and args.use_flow and not args.use_audio:
                feat = f_emd
            elif not args.use_video and not args.use_flow and args.use_audio:
                feat = audio_emd

            new_predict = mlp_cls(feat)
            new_loss = criterion(new_predict, labels)
            optim.zero_grad()
            new_loss.backward()

            index = 0
            with torch.no_grad():
                for _, p in mlp_cls.named_parameters():
                    p.sub_(eps_[index])
                    index += 1
                eps_ = eps_[index:]
                index = 0

                if args.use_video:
                    for _, p in model.module.backbone.fast_path.layer4.named_parameters():
                        p.sub_(eps_[index])
                        index += 1
                    for _, p in model.module.backbone.slow_path.layer4.named_parameters():
                        p.sub_(eps_[index])
                        index += 1
                    for name, p in model.module.cls_head.named_parameters():
                        if p.grad is None:
                            pass
                        else:
                            p.sub_(eps_[index])
                            index += 1
                    eps_ = eps_[index:]
                    index = 0
                if args.use_flow:
                    for _, p in model_flow.module.backbone.layer4.named_parameters():
                        p.sub_(eps_[index])
                        index += 1
                    for name, p in model_flow.module.cls_head.named_parameters():
                        if p.grad is None:
                            pass
                        else:
                            p.sub_(eps_[index])
                            index += 1
                    eps_ = eps_[index:]
                    index = 0
                if args.use_audio:
                    for name, p in audio_cls_model.named_parameters():
                        if p.grad is None:
                            pass
                        else:
                            p.sub_(eps_[index])
                            index += 1
            optim.step()

            return predict, loss, new_predict, new_loss
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()
            return predict, loss
    else:
        if args.use_video and args.use_flow and args.use_audio:
            a_emd_t = mlp_v2a(v_emd)
            v_emd_t = mlp_a2v(audio_emd)
            f_emd_t = mlp_v2f(v_emd)
            v_emd_t2 = mlp_f2v(f_emd)
            a_emd_t2 = mlp_f2a(f_emd)
            f_emd_t2 = mlp_a2f(audio_emd)
            a_emd_t = a_emd_t / torch.norm(a_emd_t, dim=1, keepdim=True)
            v_emd_t = v_emd_t / torch.norm(v_emd_t, dim=1, keepdim=True)
            f_emd_t = f_emd_t / torch.norm(f_emd_t, dim=1, keepdim=True)
            a_emd_t2 = a_emd_t2 / torch.norm(a_emd_t2, dim=1, keepdim=True)
            v_emd_t2 = v_emd_t2 / torch.norm(v_emd_t2, dim=1, keepdim=True)
            f_emd_t2 = f_emd_t2 / torch.norm(f_emd_t2, dim=1, keepdim=True)
            v2a_loss = torch.mean(torch.norm(a_emd_t - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
            a2v_loss = torch.mean(torch.norm(v_emd_t - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
            v2f_loss = torch.mean(torch.norm(f_emd_t - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
            f2a_loss = torch.mean(torch.norm(a_emd_t2 - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
            f2v_loss = torch.mean(torch.norm(v_emd_t2 - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
            a2f_loss = torch.mean(torch.norm(f_emd_t2 - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
            loss = loss + args.alpha_trans * (v2a_loss + a2v_loss + v2f_loss + f2a_loss + f2v_loss + a2f_loss) / 6
        elif args.use_video and args.use_flow:
            f_emd_t = mlp_v2f(v_emd)
            v_emd_t2 = mlp_f2v(f_emd)
            f_emd_t = f_emd_t / torch.norm(f_emd_t, dim=1, keepdim=True)
            v_emd_t2 = v_emd_t2 / torch.norm(v_emd_t2, dim=1, keepdim=True)
            v2f_loss = torch.mean(torch.norm(f_emd_t - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
            f2v_loss = torch.mean(torch.norm(v_emd_t2 - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
            loss = loss + args.alpha_trans * (v2f_loss + f2v_loss) / 2
        elif args.use_video and args.use_audio:
            a_emd_t = mlp_v2a(v_emd)
            v_emd_t = mlp_a2v(audio_emd)
            a_emd_t = a_emd_t / torch.norm(a_emd_t, dim=1, keepdim=True)
            v_emd_t = v_emd_t / torch.norm(v_emd_t, dim=1, keepdim=True)
            v2a_loss = torch.mean(torch.norm(a_emd_t - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
            a2v_loss = torch.mean(torch.norm(v_emd_t - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
            loss = loss + args.alpha_trans * (v2a_loss + a2v_loss) / 2
        elif args.use_flow and args.use_audio:
            a_emd_t2 = mlp_f2a(f_emd)
            f_emd_t2 = mlp_a2f(audio_emd)
            a_emd_t2 = a_emd_t2 / torch.norm(a_emd_t2, dim=1, keepdim=True)
            f_emd_t2 = f_emd_t2 / torch.norm(f_emd_t2, dim=1, keepdim=True)
            f2a_loss = torch.mean(torch.norm(a_emd_t2 - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
            a2f_loss = torch.mean(torch.norm(f_emd_t2 - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
            loss = loss + args.alpha_trans * (f2a_loss + a2f_loss) / 2

        # Supervised Contrastive Learning
        if args.use_video:
            v_emd_proj = v_proj(v_emd[:, :v_dim])
        if args.use_audio:
            a_emd_proj = a_proj(audio_emd[:, :a_dim])
        if args.use_flow:
            f_emd_proj = f_proj(f_emd[:, :f_dim])
        if args.use_video and args.use_flow and args.use_audio:
            emd_proj = torch.stack([v_emd_proj, a_emd_proj, f_emd_proj], dim=1)
        elif args.use_video and args.use_flow:
            emd_proj = torch.stack([v_emd_proj, f_emd_proj], dim=1)
        elif args.use_video and args.use_audio:
            emd_proj = torch.stack([v_emd_proj, a_emd_proj], dim=1)
        elif args.use_flow and args.use_audio:
            emd_proj = torch.stack([f_emd_proj, a_emd_proj], dim=1)

        loss_contrast = criterion_contrast(emd_proj, labels)
        loss = loss + args.alpha_contrast * loss_contrast

        # Feature Splitting with Distance
        loss_e = 0
        num_loss = 0
        if args.use_video:
            loss_e = loss_e - F.mse_loss(v_emd[:, :v_dim], v_emd[:, v_dim:])
            num_loss = num_loss + 1
        if args.use_audio:
            loss_e = loss_e - F.mse_loss(audio_emd[:, :a_dim], audio_emd[:, a_dim:])
            num_loss = num_loss + 1
        if args.use_flow:
            loss_e = loss_e - F.mse_loss(f_emd[:, :f_dim], f_emd[:, f_dim:])
            num_loss = num_loss + 1

        loss = loss + args.explore_loss_coeff * loss_e / num_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        return predict, loss


def validate_one_step(clip, labels, flow, spectrogram):
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()

    v_predict_proj = None
    f_predict_proj = None
    a_predict_proj = None
    predict1 = None
    f_predict = None
    audio_predict = None
    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = (x_slow.detach(), x_fast.detach())

            v_feat = model.module.backbone.get_predict(v_feat)
            predict1, v_emd = model.module.cls_head(v_feat)
            # v_predict_proj = v_proj_m_cls(v_proj_m(v_emd))
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)
            audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
            # a_predict_proj = a_proj_m_cls(a_proj_m(audio_emd))
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_predict, f_emd = model_flow.module.cls_head(f_feat)
            # f_predict_proj = f_proj_m_cls(f_proj_m(f_emd))

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, audio_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, audio_emd), dim=1)
        elif args.use_video and not args.use_flow and not args.use_audio:
            feat = v_emd
        elif not args.use_video and args.use_flow and not args.use_audio:
            feat = f_emd
        elif not args.use_video and not args.use_flow and args.use_audio:
            feat = audio_emd

        predict = mlp_cls(feat)

    loss = criterion(predict, labels)

    if args.SMA:
        v_predict_proj_sma = None
        f_predict_proj_sma = None
        a_predict_proj_sma = None
        predict1_sma = None
        f_predict_sma = None
        audio_predict_sma = None
        with torch.no_grad():
            if args.use_video:
                x_slow, x_fast = model_sma.module.backbone.get_feature(clip)
                v_feat = (x_slow.detach(), x_fast.detach())

                v_feat = model_sma.module.backbone.get_predict(v_feat)
                predict1_sma, v_emd = model_sma.module.cls_head(v_feat)
                # v_predict_proj_sma = v_proj_m_cls_sma(v_proj_m_sma(v_emd))
            if args.use_audio:
                _, audio_feat, _ = audio_model(spectrogram)
                audio_predict_sma, audio_emd = audio_cls_model_sma(audio_feat.detach())
                # a_predict_proj_sma = a_proj_m_cls_sma(a_proj_m_sma(audio_emd))
            if args.use_flow:
                f_feat = model_flow_sma.module.backbone.get_feature(flow)
                f_feat = model_flow_sma.module.backbone.get_predict(f_feat)
                f_predict_sma, f_emd = model_flow_sma.module.cls_head(f_feat)
                # f_predict_proj_sma = f_proj_m_cls_sma(f_proj_m_sma(f_emd))

            if args.use_video and args.use_flow and args.use_audio:
                feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
            elif args.use_video and args.use_flow:
                feat = torch.cat((v_emd, f_emd), dim=1)
            elif args.use_video and args.use_audio:
                feat = torch.cat((v_emd, audio_emd), dim=1)
            elif args.use_flow and args.use_audio:
                feat = torch.cat((f_emd, audio_emd), dim=1)
            elif args.use_video and not args.use_flow and not args.use_audio:
                feat = v_emd
            elif not args.use_video and args.use_flow and not args.use_audio:
                feat = f_emd
            elif not args.use_video and not args.use_flow and args.use_audio:
                feat = audio_emd

            predict_sma = mlp_cls_sma(feat)

        loss_sma = criterion(predict_sma, labels)
        # return predict, loss, [v_predict_proj, f_predict_proj, a_predict_proj], predict_sma, loss_sma, [v_predict_proj_sma, f_predict_proj_sma, a_predict_proj_sma]
        return predict, loss, [predict1, f_predict, audio_predict], predict_sma, loss_sma, [
            predict1_sma, f_predict_sma, audio_predict_sma]
    # return predict, loss, [v_predict_proj, f_predict_proj, a_predict_proj]
    return predict, loss, [predict1, f_predict, audio_predict]


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)


class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat


class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
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
        feat = F.normalize(self.head(feat), dim=1)
        return feat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, choices=['epic', 'hac'])
    parser.add_argument('--num_class', type=int, required=True)
    parser.add_argument('--run_name', type=str, default='')

    parser.add_argument('-s', '--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t', '--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/DATA_ROOT',
                        help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--alpha_trans', type=float, default=0.1,
                        help='alpha_trans')
    parser.add_argument("--trans_hidden_num", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temp')
    parser.add_argument('--alpha_contrast', type=float, default=3.0,
                        help='alpha_contrast')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--explore_loss_coeff', type=float, default=0.7,
                        help='explore_loss_coeff')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0,
                        help='BestAcc')
    parser.add_argument('--BestTestAcc', type=float, default=0,
                        help='BestTestAcc')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')

    parser.add_argument("--vanilla_learning", action='store_true', help='without other techniques for generalization')
    parser.add_argument("--DG_algorithm", type=str, default='None', help='naive, SAM, etc')
    parser.add_argument("--CM_mixup", action='store_true', help='mixup the representations between modalities')
    parser.add_argument("--mix_alpha", type=float, default=0.1)
    parser.add_argument("--mix_coef", type=float, default=2.0)
    parser.add_argument("--contrast", action='store_true')
    parser.add_argument("--distill_coef", type=float, default=3.0)
    parser.add_argument("--distill", action='store_true')

    parser.add_argument("--SMA", action='store_true')
    parser.add_argument("--sma_start_step", type=int, default=400)

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0'  # or 'cpu'
    device = torch.device(device)

    input_dim = 0
    num_classes = args.num_class

    cfg = None
    cfg_flow = None

    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, num_classes).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)

        v_proj = ProjectHead(input_dim=1152, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 2304

        if args.CM_mixup:
            v_proj_m = ProjectHead(input_dim=2304, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
            v_proj_m_cls = nn.Linear(128, num_classes).cuda()

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, num_classes).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        f_proj = ProjectHead(input_dim=1024, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 2048

        if args.CM_mixup:
            f_proj_m = ProjectHead(input_dim=2048, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
            f_proj_m_cls = nn.Linear(128, num_classes).cuda()

    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, num_classes)
        audio_cls_model = audio_cls_model.cuda()

        a_proj = ProjectHead(input_dim=256, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 512

        if args.CM_mixup:
            a_proj_m = ProjectHead(input_dim=512, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
            a_proj_m_cls = nn.Linear(128, num_classes).cuda()

    mlp_cls = Encoder(input_dim=input_dim, out_dim=num_classes)
    mlp_cls = mlp_cls.cuda()

    if args.use_video and args.use_flow and args.use_audio:
        mlp_v2f = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=2048).cuda()
        mlp_f2v = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=2304).cuda()
        mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304).cuda()
        mlp_f2a = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2f = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2048).cuda()
    elif args.use_video and args.use_flow:
        mlp_v2f = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=2048).cuda()
        mlp_f2v = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=2304).cuda()
    elif args.use_video and args.use_audio:
        mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304).cuda()
    elif args.use_flow and args.use_audio:
        mlp_f2a = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2f = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2048).cuda()


    modal_code = _build_modality_code(args.use_video, args.use_audio, args.use_flow)
    log_name = build_run_name(METHOD_NAME, DATASET_NAME, args.source_domain, args.target_domain,
                              modal_code, args.seed, args.appen, args.run_name)
    log_dir, model_dir, log_path = build_output_paths(
        __file__, DATASET_NAME, METHOD_NAME, args.dg_mode, log_name
    )
    print('Log path:', log_path)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    batch_size = args.bsz

    criterion_contrast = SupConLoss(temperature=args.temp)
    criterion_contrast = criterion_contrast.cuda()

    params = list(mlp_cls.parameters())
    if not args.vanilla_learning:
        if args.use_video:
            params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
                model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(
                v_proj.parameters())

        if args.use_flow:
            params = params + list(model_flow.module.backbone.layer4.parameters()) + list(
                model_flow.module.cls_head.parameters()) + list(f_proj.parameters())

        if args.use_audio:
            params = params + list(audio_cls_model.parameters()) + list(a_proj.parameters())
            if args.CM_mixup:
                params = params + list(a_proj_m.parameters()) + list(a_proj_m_cls.parameters())

        if args.use_video and args.use_flow and args.use_audio:
            params = params + list(mlp_v2a.parameters()) + list(mlp_a2v.parameters())
            params = params + list(mlp_v2f.parameters()) + list(mlp_f2v.parameters())
            params = params + list(mlp_f2a.parameters()) + list(mlp_a2f.parameters())
        elif args.use_video and args.use_flow:
            params = params + list(mlp_v2f.parameters()) + list(mlp_f2v.parameters())
        elif args.use_video and args.use_audio:
            params = params + list(mlp_v2a.parameters()) + list(mlp_a2v.parameters())
        elif args.use_flow and args.use_audio:
            params = params + list(mlp_f2a.parameters()) + list(mlp_a2f.parameters())
    else:
        if args.use_video:
            params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
                model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters())
            if args.CM_mixup:
                params = params + list(v_proj_m.parameters()) + list(v_proj_m_cls.parameters())
        if args.use_flow:
            params = params + list(model_flow.module.backbone.layer4.parameters()) + list(
                model_flow.module.cls_head.parameters())
            if args.CM_mixup:
                params = params + list(f_proj_m.parameters()) + list(f_proj_m_cls.parameters())
        if args.use_audio:
            params = params + list(audio_cls_model.parameters())
            if args.CM_mixup:
                params = params + list(a_proj_m.parameters()) + list(a_proj_m_cls.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc

    if args.resume:
        resume_file = os.path.join(model_dir, log_name + '.pt')
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch'] + 1

        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']
        BestTestAcc = checkpoint['BestTestAcc']

        if args.use_video:
            model.load_state_dict(checkpoint['model_state_dict'])
            v_proj.load_state_dict(checkpoint['v_proj_state_dict'])
        if args.use_flow:
            model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
            f_proj.load_state_dict(checkpoint['f_proj_state_dict'])
        if args.use_audio:
            audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
            audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
            a_proj.load_state_dict(checkpoint['a_proj_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        if args.use_video and args.use_flow and args.use_audio:
            mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
            mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
            mlp_v2f.load_state_dict(checkpoint['mlp_v2f_state_dict'])
            mlp_f2v.load_state_dict(checkpoint['mlp_f2v_state_dict'])
            mlp_f2a.load_state_dict(checkpoint['mlp_f2a_state_dict'])
            mlp_a2f.load_state_dict(checkpoint['mlp_a2f_state_dict'])
        elif args.use_video and args.use_flow:
            mlp_v2f.load_state_dict(checkpoint['mlp_v2f_state_dict'])
            mlp_f2v.load_state_dict(checkpoint['mlp_f2v_state_dict'])
        elif args.use_video and args.use_audio:
            mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
            mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
        elif args.use_flow and args.use_audio:
            mlp_f2a.load_state_dict(checkpoint['mlp_f2a_state_dict'])
            mlp_a2f.load_state_dict(checkpoint['mlp_a2f_state_dict'])
        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
    else:
        print("Training From Scratch ...")
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)


    train_dataset = DatasetClass(**make_dataset_kwargs(args, 'train', args.source_domain, cfg, cfg_flow, source=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=(device.type == 'cuda'), drop_last=True)
    validate_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.source_domain, cfg, cfg_flow, source=True))
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=(device.type == 'cuda'), drop_last=False)
    test_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain, cfg, cfg_flow, source=False))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                                  shuffle=False,
                                                  pin_memory=(device.type == 'cuda'), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader, 'test': test_dataloader}
    global_step = starting_epoch * len(train_dataloader)
    sma_count = 0
    with open(log_path, "a") as f:
        write_all_hparams(f, args)
        if args.SMA:
            if args.use_video:
                model_sma = copy.deepcopy(model)
                model_sma.eval()
                v_proj_m_sma = copy.deepcopy(v_proj_m)
                v_proj_m_sma.eval()
                v_proj_m_cls_sma = copy.deepcopy(v_proj_m_cls)
                v_proj_m_cls_sma.eval()
            if args.use_flow:
                model_flow_sma = copy.deepcopy(model_flow)
                model_flow_sma.eval()
                f_proj_m_sma = copy.deepcopy(f_proj_m)
                f_proj_m_sma.eval()
                f_proj_m_cls_sma = copy.deepcopy(f_proj_m_cls)
                f_proj_m_cls_sma.eval()
            if args.use_audio:
                audio_cls_model_sma = copy.deepcopy(audio_cls_model)
                audio_cls_model_sma.eval()
                a_proj_m_sma = copy.deepcopy(a_proj_m)
                a_proj_m_sma.eval()
                a_proj_m_cls_sma = copy.deepcopy(a_proj_m_cls)
                a_proj_m_cls_sma.eval()
            mlp_cls_sma = copy.deepcopy(mlp_cls)
            mlp_cls_sma.eval()
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                total_new_loss = 0
                new_acc = 0
                acc_sma = 0

                acc_avg = 0
                acc_avg_sma = 0
                print(split)
                mlp_cls.train(split == 'train')
                if args.use_video:
                    model.train(split == 'train')
                    if args.CM_mixup:
                        v_proj_m.train(split == 'train')
                        v_proj_m_cls.train(split == 'train')
                    if not args.vanilla_learning:
                        v_proj.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                    if args.CM_mixup:
                        f_proj_m.train(split == 'train')
                        f_proj_m_cls.train(split == 'train')
                    if not args.vanilla_learning:
                        f_proj.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
                    if args.CM_mixup:
                        a_proj_m.train(split == 'train')
                        a_proj_m_cls.train(split == 'train')
                    if not args.vanilla_learning:
                        a_proj.train(split == 'train')
                if args.use_video and args.use_flow and args.use_audio:
                    if not args.vanilla_learning:
                        mlp_v2a.train(split == 'train')
                        mlp_a2v.train(split == 'train')
                        mlp_v2f.train(split == 'train')
                        mlp_f2v.train(split == 'train')
                        mlp_f2a.train(split == 'train')
                        mlp_a2f.train(split == 'train')
                elif args.use_video and args.use_flow:
                    if not args.vanilla_learning:
                        mlp_v2f.train(split == 'train')
                        mlp_f2v.train(split == 'train')
                elif args.use_video and args.use_audio:
                    if not args.vanilla_learning:
                        mlp_v2a.train(split == 'train')
                        mlp_a2v.train(split == 'train')
                elif args.use_flow and args.use_audio:
                    if not args.vanilla_learning:
                        mlp_f2a.train(split == 'train')
                        mlp_a2f.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, flow, spectrogram, labels)) in enumerate(dataloaders[split]):
                        if split == 'train':
                            global_step += 1
                            if args.vanilla_learning and args.DG_algorithm == 'SAM':
                                predict1, loss, new_predict1, new_loss = train_one_step(clip, labels, flow, spectrogram, global_step)
                            else:
                                predict1, loss = train_one_step(clip, labels, flow, spectrogram, global_step)

                            if args.SMA:
                                new_video_dict = {}
                                new_flow_dict = {}
                                new_audio_dict = {}
                                new_v_proj_m_dict = {}
                                new_f_proj_m_dict = {}
                                new_a_proj_m_dict = {}
                                new_v_proj_m_cls_dict = {}
                                new_f_proj_m_cls_dict = {}
                                new_a_proj_m_cls_dict = {}
                                new_cls_dict = {}
                                if global_step > args.sma_start_step:
                                    sma_count += 1

                                    if args.use_video:
                                        for (name, param_q), (_, param_k) in zip(model.state_dict().items(),
                                                                                 model_sma.state_dict().items()):
                                            new_video_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                        for (name, param_q), (_, param_k) in zip(v_proj_m.state_dict().items(),
                                                                                 v_proj_m_sma.state_dict().items()):
                                            new_v_proj_m_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                        for (name, param_q), (_, param_k) in zip(v_proj_m_cls.state_dict().items(),
                                                                                 v_proj_m_cls_sma.state_dict().items()):
                                            new_v_proj_m_cls_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                    if args.use_flow:
                                        for (name, param_q), (_, param_k) in zip(model_flow.state_dict().items(),
                                                                                 model_flow_sma.state_dict().items()):
                                            new_flow_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                               1. + sma_count))
                                        for (name, param_q), (_, param_k) in zip(f_proj_m.state_dict().items(),
                                                                                 f_proj_m_sma.state_dict().items()):
                                            new_f_proj_m_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                        for (name, param_q), (_, param_k) in zip(f_proj_m_cls.state_dict().items(),
                                                                                 f_proj_m_cls_sma.state_dict().items()):
                                            new_f_proj_m_cls_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                    if args.use_audio:
                                        for (name, param_q), (_, param_k) in zip(audio_cls_model.state_dict().items(),
                                                                                 audio_cls_model_sma.state_dict().items()):
                                            new_audio_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                        for (name, param_q), (_, param_k) in zip(a_proj_m.state_dict().items(),
                                                                                 a_proj_m_sma.state_dict().items()):
                                            new_a_proj_m_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                        for (name, param_q), (_, param_k) in zip(a_proj_m_cls.state_dict().items(),
                                                                                 a_proj_m_cls_sma.state_dict().items()):
                                            new_a_proj_m_cls_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                                1. + sma_count))
                                    for (name, param_q), (_, param_k) in zip(mlp_cls.state_dict().items(),
                                                                             mlp_cls_sma.state_dict().items()):
                                        new_cls_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                                                      1. + sma_count))
                                else:
                                    if args.use_video:
                                        for (name, param_q), (_, param_k) in zip(model.state_dict().items(),
                                                                                 model_sma.state_dict().items()):
                                            new_video_dict[name] = param_q.detach().data.clone()
                                        for (name, param_q), (_, param_k) in zip(v_proj_m.state_dict().items(),
                                                                                 v_proj_m_sma.state_dict().items()):
                                            new_v_proj_m_dict[name] = param_q.detach().data.clone()
                                        for (name, param_q), (_, param_k) in zip(v_proj_m_cls.state_dict().items(),
                                                                                 v_proj_m_cls_sma.state_dict().items()):
                                            new_v_proj_m_cls_dict[name] = param_q.detach().data.clone()
                                    if args.use_flow:
                                        for (name, param_q), (_, param_k) in zip(model_flow.state_dict().items(),
                                                                                 model_flow_sma.state_dict().items()):
                                            new_flow_dict[name] = param_q.detach().data.clone()
                                        for (name, param_q), (_, param_k) in zip(f_proj_m.state_dict().items(),
                                                                                 f_proj_m_sma.state_dict().items()):
                                            new_f_proj_m_dict[name] = param_q.detach().data.clone()
                                        for (name, param_q), (_, param_k) in zip(f_proj_m_cls.state_dict().items(),
                                                                                 f_proj_m_cls_sma.state_dict().items()):
                                            new_f_proj_m_cls_dict[name] = param_q.detach().data.clone()
                                    if args.use_audio:
                                        for (name, param_q), (_, param_k) in zip(audio_cls_model.state_dict().items(),
                                                                                 audio_cls_model_sma.state_dict().items()):
                                            new_audio_dict[name] = param_q.detach().data.clone()
                                        for (name, param_q), (_, param_k) in zip(a_proj_m.state_dict().items(),
                                                                                 a_proj_m_sma.state_dict().items()):
                                            new_a_proj_m_dict[name] = param_q.detach().data.clone()
                                        for (name, param_q), (_, param_k) in zip(a_proj_m_cls.state_dict().items(),
                                                                                 a_proj_m_cls_sma.state_dict().items()):
                                            new_a_proj_m_cls_dict[name] = param_q.detach().data.clone()
                                    for (name, param_q), (_, param_k) in zip(mlp_cls.state_dict().items(),
                                                                             mlp_cls_sma.state_dict().items()):
                                        new_cls_dict[name] = param_q.detach().data.clone()

                                if args.use_video:
                                    model_sma.load_state_dict(new_video_dict)
                                    v_proj_m_sma.load_state_dict(new_v_proj_m_dict)
                                    v_proj_m_cls_sma.load_state_dict(new_v_proj_m_cls_dict)
                                if args.use_flow:
                                    model_flow_sma.load_state_dict(new_flow_dict)
                                    f_proj_m_sma.load_state_dict(new_f_proj_m_dict)
                                    f_proj_m_cls_sma.load_state_dict(new_f_proj_m_cls_dict)
                                if args.use_audio:
                                    audio_cls_model_sma.load_state_dict(new_audio_dict)
                                    a_proj_m_sma.load_state_dict(new_a_proj_m_dict)
                                    a_proj_m_cls_sma.load_state_dict(new_a_proj_m_cls_dict)
                                mlp_cls_sma.load_state_dict(new_cls_dict)
                        else:
                            if args.use_video:
                                model.eval()
                                if args.CM_mixup:
                                    v_proj_m.eval()
                                    v_proj_m_cls.eval()
                            if args.use_flow:
                                model_flow.eval()
                                if args.CM_mixup:
                                    f_proj_m.eval()
                                    f_proj_m_cls.eval()
                            if args.use_audio:
                                audio_cls_model.eval()
                                if args.CM_mixup:
                                    a_proj_m.eval()
                                    a_proj_m_cls.eval()
                            if args.SMA:
                                predict1, loss, unimodal_predict, predict1_sma, loss_sma, unimodal_sma_predict = validate_one_step(clip, labels, flow,
                                                                                           spectrogram)
                                video_predict = unimodal_predict[0]
                                flow_predict = unimodal_predict[1]
                                audio_predict = unimodal_predict[2]

                                video_predict_sma = unimodal_sma_predict[0]
                                flow_predict_sma = unimodal_sma_predict[1]
                                audio_predict_sma = unimodal_sma_predict[2]

                                _, predict_sma = torch.max(predict1_sma.detach().cpu(), dim=1)
                                acc1_sma = (predict_sma == labels).sum().item()
                                acc_sma += int(acc1_sma)

                                predict1_avg = copy.deepcopy(predict1)
                                predict1_sma_avg = copy.deepcopy(predict1_sma)
                                modality_use_number = 0
                                if args.use_video:
                                    modality_use_number += 1
                                    predict1_avg = predict1_avg + video_predict
                                    predict1_sma_avg = predict1_sma_avg + video_predict_sma
                                if args.use_flow:
                                    modality_use_number += 1
                                    predict1_avg = predict1_avg + flow_predict
                                    predict1_sma_avg = predict1_sma_avg + flow_predict_sma
                                if args.use_audio:
                                    modality_use_number += 1
                                    predict1_avg = predict1_avg + audio_predict
                                    predict1_sma_avg = predict1_sma_avg + audio_predict_sma
                                predict1_avg /= (modality_use_number+1)
                                predict1_sma_avg /= (modality_use_number + 1)

                                _, predict_avg = torch.max(predict1_avg.detach().cpu(), dim=1)
                                acc1_avg = (predict_avg == labels).sum().item()
                                acc_avg += int(acc1_avg)
                                _, predict_sma_avg = torch.max(predict1_sma_avg.detach().cpu(), dim=1)
                                acc1_sma_avg = (predict_sma_avg == labels).sum().item()
                                acc_avg_sma += int(acc1_sma_avg)
                            else:
                                predict1, loss, unimodal_predict = validate_one_step(clip, labels, flow, spectrogram)
                                video_predict = unimodal_predict[0]
                                flow_predict = unimodal_predict[1]
                                audio_predict = unimodal_predict[2]

                                predict1_avg = copy.deepcopy(predict1)
                                modality_use_number = 0
                                if args.use_video:
                                    modality_use_number += 1
                                    predict1_avg = predict1_avg + video_predict
                                if args.use_flow:
                                    modality_use_number += 1
                                    predict1_avg = predict1_avg + flow_predict
                                if args.use_audio:
                                    modality_use_number += 1
                                    predict1_avg = predict1_avg + audio_predict
                                predict1_avg /= (modality_use_number + 1)

                                _, predict_avg = torch.max(predict1_avg.detach().cpu(), dim=1)
                                acc1_avg = (predict_avg == labels).sum().item()
                                acc_avg += int(acc1_avg)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        if args.vanilla_learning and args.DG_algorithm == 'SAM' and split == 'train':
                            total_new_loss += new_loss.item() * batch_size
                            _, new_predict = torch.max(new_predict1.detach().cpu(), dim=1)
                            new_acc1 = (new_predict == labels).sum().item()
                            new_acc += int(new_acc1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(
                                total_loss / float(count),
                                loss.item(),
                                acc / float(count)))
                        pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)

                    if split == 'test':
                        currenttestAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestTestAcc = currenttestAcc
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()

                                if args.use_video:
                                    save['v_proj_state_dict'] = v_proj.state_dict()
                                    save['model_state_dict'] = model.state_dict()
                                if args.use_flow:
                                    save['f_proj_state_dict'] = f_proj.state_dict()
                                    save['model_flow_state_dict'] = model_flow.state_dict()
                                if args.use_audio:
                                    save['a_proj_state_dict'] = a_proj.state_dict()
                                    save['audio_model_state_dict'] = audio_model.state_dict()
                                    save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                                if args.use_video and args.use_flow and args.use_audio:
                                    save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                    save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                    save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                    save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                    save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                    save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()
                                elif args.use_video and args.use_flow:
                                    save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                    save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                elif args.use_video and args.use_audio:
                                    save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                    save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                elif args.use_flow and args.use_audio:
                                    save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                    save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()

                                torch.save(save, os.path.join(model_dir, log_name + '_best.pt'))

                        if args.save_checkpoint:
                            save = {
                                'epoch': epoch_i,
                                'BestLoss': BestLoss,
                                'BestEpoch': BestEpoch,
                                'BestAcc': BestAcc,
                                'BestTestAcc': BestTestAcc,
                                'optimizer': optim.state_dict(),
                            }
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()

                            if args.use_video:
                                save['v_proj_state_dict'] = v_proj.state_dict()
                                save['model_state_dict'] = model.state_dict()
                            if args.use_flow:
                                save['f_proj_state_dict'] = f_proj.state_dict()
                                save['model_flow_state_dict'] = model_flow.state_dict()
                            if args.use_audio:
                                save['a_proj_state_dict'] = a_proj.state_dict()
                                save['audio_model_state_dict'] = audio_model.state_dict()
                                save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                            if args.use_video and args.use_flow and args.use_audio:
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()
                            elif args.use_video and args.use_flow:
                                save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                            elif args.use_video and args.use_audio:
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                            elif args.use_flow and args.use_audio:
                                save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()

                            torch.save(save, os.path.join(model_dir, log_name + '.pt'))

                    if args.vanilla_learning and args.DG_algorithm == 'SAM' and not args.SMA:
                        f.write(
                            "{},{},{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count),
                                                         total_new_loss / float(count), new_acc / float(count)))
                    elif args.SMA and split != 'train':
                        f.write("{},{},{},{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count),
                                                          acc_sma / float(count), acc_avg / float(count), acc_avg_sma / float(count)))
                    else:
                        f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count), acc_avg / float(count),))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)

                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch,
                                                                                                         BestLoss,
                                                                                                         BestAcc,
                                                                                                         BestTestAcc))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc,
                                                                                  BestTestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)

    f.close()
