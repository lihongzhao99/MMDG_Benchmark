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
from torch.distributions import Categorical
import itertools


METHOD_NAME = 'MOOSA'


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

# cpu_num = 1
# # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# os.environ["OMP_NUM_THREADS"] = str(cpu_num)
# os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
# os.environ["MKL_NUM_THREADS"] = str(cpu_num)
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
# os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
# torch.set_num_threads(cpu_num)
# cv2.setNumThreads(cpu_num)	# 0也可以
# cv2.ocl.setUseOpenCL(False)


def train_one_step(clip, labels, flow, spectrogram):
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
        v_feat = model.module.backbone.get_predict(v_feat)
        predict1, v_emd = model.module.cls_head(v_feat)
        v_dim = int(v_emd.shape[1] / 2)
        entropyp = Categorical(probs = nn.Softmax(dim=1)(predict1)).entropy().reshape(-1,1)
        output_loss1 = criterion(predict1, labels)
        video_parts = torch.split(v_emd, v_emd.shape[1] // args.jigsaw_num_splits, dim=1)
        output_u_v = F.softmax(predict1, 1)
        loss_u_v = (-output_u_v * torch.log(output_u_v + 1e-5)).sum(1).mean()

    if args.use_flow:
        f_feat = model_flow.module.backbone.get_predict(f_feat.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat)
        f_dim = int(f_emd.shape[1] / 2)
        entropyf = Categorical(probs = nn.Softmax(dim=1)(f_predict)).entropy().reshape(-1,1)
        output_loss3 = criterion(f_predict, labels)
        flow_parts = torch.split(f_emd, f_emd.shape[1] // args.jigsaw_num_splits, dim=1)
        output_u_f = F.softmax(f_predict, 1)
        loss_u_f = (-output_u_f * torch.log(output_u_f + 1e-5)).sum(1).mean()

    if args.use_audio:    
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
        a_dim = int(audio_emd.shape[1] / 2)
        entropya = Categorical(probs = nn.Softmax(dim=1)(audio_predict)).entropy().reshape(-1,1)
        output_loss2 = criterion(audio_predict, labels)
        audio_parts = torch.split(audio_emd, audio_emd.shape[1] // args.jigsaw_num_splits, dim=1)
        output_u_a = F.softmax(audio_predict, 1)
        loss_u_a = (-output_u_a * torch.log(output_u_a + 1e-5)).sum(1).mean()

    if args.use_video and args.use_flow and args.use_audio:
        feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
        parts = video_parts + audio_parts + flow_parts
    elif args.use_video and args.use_flow:
        feat = torch.cat((v_emd, f_emd), dim=1)
        parts = video_parts + flow_parts
    elif args.use_video and args.use_audio:
        feat = torch.cat((v_emd, audio_emd), dim=1)
        parts = video_parts + audio_parts 
    elif args.use_flow and args.use_audio:
        feat = torch.cat((f_emd, audio_emd), dim=1)
        parts = flow_parts + audio_parts

    predict = mlp_cls(feat)
    output_loss4 = criterion(predict, labels)
    entropypa = Categorical(probs = nn.Softmax(dim=1)(predict)).entropy().reshape(-1,1)
    output_u = F.softmax(predict, 1)
    loss_u = (-output_u * torch.log(output_u + 1e-5)).sum(1).mean()

    # Entropy Weighting
    if args.use_video and args.use_flow and args.use_audio:
        entropy = -torch.cat((entropyp, entropya, entropyf, entropypa), 1)
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
        loss = torch.mean(output[:,0]*output_loss1+output[:,1]*output_loss2+output[:,2]*output_loss3+output[:,3]*output_loss4)
        loss_ent_min = (loss_u + loss_u_v + loss_u_a + loss_u_f) * args.entropy_min_weight / 4
    elif args.use_video and args.use_flow:
        entropy = -torch.cat((entropyp, entropyf, entropypa), 1)
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
        loss = torch.mean(output[:,0]*output_loss1+output[:,1]*output_loss3+output[:,2]*output_loss4)
        loss_ent_min = (loss_u + loss_u_v + loss_u_f) * args.entropy_min_weight / 3
    elif args.use_video and args.use_audio:
        entropy = -torch.cat((entropyp, entropya, entropypa), 1)
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
        loss = torch.mean(output[:,0]*output_loss1+output[:,1]*output_loss2+output[:,2]*output_loss4)
        loss_ent_min = (loss_u + loss_u_v + loss_u_a) * args.entropy_min_weight / 3
    elif args.use_flow and args.use_audio:
        entropy = -torch.cat((entropyf, entropya, entropypa), 1)
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
        loss = torch.mean(output[:,0]*output_loss3+output[:,1]*output_loss2+output[:,2]*output_loss4)
        loss_ent_min = (loss_u + loss_u_a + loss_u_f) * args.entropy_min_weight / 3

    # Multimodal Jigsaw Puzzles
    all_combinations = list(itertools.permutations(parts, len(parts)))
    all_combinations = [all_combinations[ji] for ji in jigsaw_indices]
    jigsaw_labels = []
    combinations = []
    for label, all_parts in enumerate(all_combinations):
        concatenated = torch.cat(all_parts, dim=1)
        jigsaw_labels.append(torch.tensor([label]).repeat(concatenated.shape[0], 1))
        combinations.append(concatenated)
    combinations = torch.cat(combinations, dim=0)
    jigsaw_labels = torch.cat(jigsaw_labels, dim=0).squeeze(1).type(torch.LongTensor).cuda()
    predict_jigsaw = jigsaw_cls(combinations)
    loss_jigsaw = nn.CrossEntropyLoss()(predict_jigsaw, jigsaw_labels)
    loss = loss + loss_jigsaw*args.jigsaw_ratio

    # Entropy Minimization
    loss = loss + loss_ent_min

    # Masked Cross-modal Translation 
    if args.use_video and args.use_flow and args.use_audio:
        mask_v = torch.rand_like(v_emd) < args.mask_ratio
        v_emd_masked = v_emd.clone()  
        v_emd_masked[mask_v] = 0 

        mask_a = torch.rand_like(audio_emd) < args.mask_ratio
        audio_emd_masked = audio_emd.clone()  
        audio_emd_masked[mask_a] = 0 

        mask_f = torch.rand_like(f_emd) < args.mask_ratio
        f_emd_masked = f_emd.clone()  
        f_emd_masked[mask_f] = 0 

        a_emd_t = mlp_v2a(v_emd_masked)
        v_emd_t = mlp_a2v(audio_emd_masked)
        f_emd_t = mlp_v2f(v_emd_masked)
        v_emd_t2 = mlp_f2v(f_emd_masked)
        a_emd_t2 = mlp_f2a(f_emd_masked)
        f_emd_t2 = mlp_a2f(audio_emd_masked)
        a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
        f_emd_t = f_emd_t/torch.norm(f_emd_t, dim=1, keepdim=True)
        a_emd_t2 = a_emd_t2/torch.norm(a_emd_t2, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2/torch.norm(v_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2/torch.norm(f_emd_t2, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        v2f_loss = torch.mean(torch.norm(f_emd_t-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2a_loss = torch.mean(torch.norm(a_emd_t2-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2a_loss + a2v_loss+ v2f_loss+ f2a_loss+ f2v_loss+ a2f_loss)/6
    elif args.use_video and args.use_flow:
        mask_v = torch.rand_like(v_emd) < args.mask_ratio
        v_emd_masked = v_emd.clone()  
        v_emd_masked[mask_v] = 0 

        mask_f = torch.rand_like(f_emd) < args.mask_ratio
        f_emd_masked = f_emd.clone()  
        f_emd_masked[mask_f] = 0 

        f_emd_t = mlp_v2f(v_emd_masked)
        v_emd_t2 = mlp_f2v(f_emd_masked)
        f_emd_t = f_emd_t/torch.norm(f_emd_t, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2/torch.norm(v_emd_t2, dim=1, keepdim=True)
        v2f_loss = torch.mean(torch.norm(f_emd_t-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2f_loss+ f2v_loss)/2
    elif args.use_video and args.use_audio:
        mask_v = torch.rand_like(v_emd) < args.mask_ratio
        v_emd_masked = v_emd.clone()  
        v_emd_masked[mask_v] = 0 

        mask_a = torch.rand_like(audio_emd) < args.mask_ratio
        audio_emd_masked = audio_emd.clone()  
        audio_emd_masked[mask_a] = 0 

        a_emd_t = mlp_v2a(v_emd_masked)
        v_emd_t = mlp_a2v(audio_emd_masked)
        a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2a_loss + a2v_loss)/2
    elif args.use_flow and args.use_audio:
        mask_a = torch.rand_like(audio_emd) < args.mask_ratio
        audio_emd_masked = audio_emd.clone()  
        audio_emd_masked[mask_a] = 0 

        mask_f = torch.rand_like(f_emd) < args.mask_ratio
        f_emd_masked = f_emd.clone()  
        f_emd_masked[mask_f] = 0 

        a_emd_t2 = mlp_f2a(f_emd_masked)
        f_emd_t2 = mlp_a2f(audio_emd_masked)
        a_emd_t2 = a_emd_t2/torch.norm(a_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2/torch.norm(f_emd_t2, dim=1, keepdim=True)
        f2a_loss = torch.mean(torch.norm(a_emd_t2-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(f2a_loss + a2f_loss)/2

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
    loss = loss + args.alpha_contrast*loss_contrast
  
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
    
    loss = loss + args.explore_loss_coeff * loss_e/num_loss

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
    
    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip) 
            v_feat = (x_slow.detach(), x_fast.detach())  

            v_feat = model.module.backbone.get_predict(v_feat)
            predict1, v_emd = model.module.cls_head(v_feat)
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)
            audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)  
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_predict, f_emd = model_flow.module.cls_head(f_feat)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, audio_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, audio_emd), dim=1)

        predict = mlp_cls(feat)

    loss = criterion(predict, labels)

    return predict, loss

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

class EncoderJigsaw(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderJigsaw, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)

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

    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
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
    parser.add_argument('--resumef', action='store_true')
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
    parser.add_argument('--entropy_weight_temp', type=float, default=1.0,
                        help='entropy_weight_temp')
    parser.add_argument('--entropy_min_weight', type=float, default=0.001,
                        help='entropy_min_weight')
    parser.add_argument('--jigsaw_ratio', type=float, default=1.0,
                        help='jigsaw_ratio')
    parser.add_argument("--jigsaw_num_splits", type=int, default=4)
    parser.add_argument("--jigsaw_samples", type=int, default=128)
    parser.add_argument("--jigsaw_hidden", type=int, default=512)
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                        help='mask_ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
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

    if args.use_video and args.use_flow and args.use_audio:
        jigsaw_indices = random.sample(range(np.math.factorial(3*args.jigsaw_num_splits)), args.jigsaw_samples)
    else:
        jigsaw_indices = random.sample(range(np.math.factorial(2*args.jigsaw_num_splits)), args.jigsaw_samples)
    print('jigsaw_indices: ', jigsaw_indices)

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
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

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, num_classes).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        f_proj = ProjectHead(input_dim=1024, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 2048

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

    mlp_cls = Encoder(input_dim=input_dim, out_dim=num_classes)
    mlp_cls = mlp_cls.cuda()

    jigsaw_cls = EncoderJigsaw(input_dim=input_dim, out_dim=args.jigsaw_samples, hidden=args.jigsaw_hidden)
    jigsaw_cls = jigsaw_cls.cuda()

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
    if args.use_video:
        params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(v_proj.parameters())
    if args.use_flow:
        params = params + list(model_flow.module.backbone.layer4.parameters()) +list(model_flow.module.cls_head.parameters()) + list(f_proj.parameters())
    if args.use_audio:
        params = params + list(audio_cls_model.parameters()) + list(a_proj.parameters())
    
    if args.use_video and args.use_flow and args.use_audio:
        params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())
        params = params + list(mlp_v2f.parameters())+list(mlp_f2v.parameters())
        params = params + list(mlp_f2a.parameters())+list(mlp_a2f.parameters())
    elif args.use_video and args.use_flow:
        params = params + list(mlp_v2f.parameters())+list(mlp_f2v.parameters())
    elif args.use_video and args.use_audio:
        params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())
    elif args.use_flow and args.use_audio:
        params = params + list(mlp_f2a.parameters())+list(mlp_a2f.parameters())

    params = params + list(jigsaw_cls.parameters())
    
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
   
    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc

    if args.resumef:
        resume_file = os.path.join(model_dir, log_name + '.pt')
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
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
        print("Training From Scratch ..." )
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
    with open(log_path, "a") as f:
        write_all_hparams(f, args)
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                mlp_cls.train(split == 'train')
                if args.use_video:
                    model.train(split == 'train')
                    v_proj.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                    f_proj.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
                    a_proj.train(split == 'train')
                if args.use_video and args.use_flow and args.use_audio:
                    mlp_v2a.train(split == 'train')
                    mlp_a2v.train(split == 'train')
                    mlp_v2f.train(split == 'train')
                    mlp_f2v.train(split == 'train')
                    mlp_f2a.train(split == 'train')
                    mlp_a2f.train(split == 'train')
                elif args.use_video and args.use_flow:
                    mlp_v2f.train(split == 'train')
                    mlp_f2v.train(split == 'train')
                elif args.use_video and args.use_audio:
                    mlp_v2a.train(split == 'train')
                    mlp_a2v.train(split == 'train')
                elif args.use_flow and args.use_audio:
                    mlp_f2a.train(split == 'train')
                    mlp_a2f.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, flow, spectrogram, labels)) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(clip, labels, flow, spectrogram)
                        else:
                            predict1, loss = validate_one_step(clip, labels, flow, spectrogram)

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

                                torch.save(save, os.path.join(model_dir, log_name + '_best_{}.pt'.format(epoch_i)))

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
                        
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)

    f.close()
