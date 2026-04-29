import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm

import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloaders.dataloader_EPIC_domain_labels import EPICDOMAIN
from dataloaders.dataloader_HAC_domain_labels import HACDOMAIN
import torch.nn.functional as F
from datetime import datetime


METHOD_NAME = 'JAT'


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


def build_checkpoint(epoch_i):
    save = {
        'epoch': epoch_i,
        'BestLoss': BestLoss,
        'BestEpoch': BestEpoch,
        'BestAcc': BestAcc,
        'BestTestAcc': BestTestAcc,
        'optimizer': optim.state_dict(),
        'mlp_cls_state_dict': mlp_cls.state_dict(),
        'modal_disc_state_dict': modal_disc.state_dict(),
        'domain_disc_state_dict': domain_disc.state_dict(),
        'domain_disc_global_state_dict': domain_disc_global.state_dict(),
    }
    if args.use_video:
        save['model_state_dict'] = model.state_dict()
        save['v_proj_state_dict'] = v_proj.state_dict()
        save['v_proj2_state_dict'] = v_proj2.state_dict()
    if args.use_flow:
        save['model_flow_state_dict'] = model_flow.state_dict()
        save['f_proj_state_dict'] = f_proj.state_dict()
        save['f_proj2_state_dict'] = f_proj2.state_dict()
    if args.use_audio:
        save['audio_model_state_dict'] = audio_model.state_dict()
        save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
        save['a_proj_state_dict'] = a_proj.state_dict()
        save['a_proj2_state_dict'] = a_proj2.state_dict()
    return save


def _entropy_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()

def train_one_step(clip, labels, flow, spectrogram, domain_labels):
    labels = labels.cuda()
    domain_labels = domain_labels.cuda()
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


    domain_adv_loss_local = 0
    modal_adv_loss = 0
    cls_loss = 0

    weights = {}
    entropies = {}


    modal_index = 0
    if args.use_video:
        video_labels = modal_index * torch.ones(batch_size, dtype=torch.long, device='cuda')
        modal_index += 1

    if args.use_audio:
        audio_labels = modal_index * torch.ones(batch_size, dtype=torch.long, device='cuda')
        modal_index += 1

    if args.use_flow:
        flow_labels = modal_index * torch.ones(batch_size, dtype=torch.long, device='cuda')
        modal_index += 1

    if args.use_video:
        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)
        v_loss = criterion(v_predict, labels)


        v_emd_rev = grad_reverse(v_emd, alpha=args.alpha_rev)
        v_emd_proj = v_proj(v_emd_rev)
        if args.enable_domain_adv:

            v_emd_rev2 = grad_reverse(v_emd, alpha=args.alpha_rev2)
            v_emd_proj2 = v_proj2(v_emd_rev2)


            v_domain_predict = domain_disc(v_emd_proj2)
            v_domain_loss = criterion(v_domain_predict, domain_labels)
            entropies['v'] = _entropy_from_logits(v_domain_predict)
            domain_adv_loss_local += v_domain_loss
        else:
            entropies['v'] = _entropy_from_logits(v_predict)


        v_modal_predict = modal_disc(v_emd_proj)
        v_modal_loss = criterion(v_modal_predict, video_labels)
        modal_adv_loss += v_modal_loss


    if args.use_flow:

        f_feat = model_flow.module.backbone.get_predict(f_feat)
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

        f_loss = criterion(f_predict, labels)


        f_emd_rev = grad_reverse(f_emd, alpha=args.alpha_rev)
        f_emd_proj = f_proj(f_emd_rev)
        if args.enable_domain_adv:

            f_emd_rev2 = grad_reverse(f_emd, alpha=args.alpha_rev2)
            f_emd_proj2 = f_proj2(f_emd_rev2)


            f_domain_predict = domain_disc(f_emd_proj2)
            f_domain_loss = criterion(f_domain_predict, domain_labels)
            entropies['f'] = _entropy_from_logits(f_domain_predict)
            domain_adv_loss_local += f_domain_loss
        else:
            entropies['f'] = _entropy_from_logits(f_predict)


        f_modal_predict = modal_disc(f_emd_proj)
        f_modal_loss = criterion(f_modal_predict, flow_labels)
        modal_adv_loss += f_modal_loss

    if args.use_audio:

        a_predict, audio_emd = audio_cls_model(audio_feat)

        a_loss = criterion(a_predict, labels)


        a_emd_rev = grad_reverse(audio_emd, alpha=args.alpha_rev)
        a_emd_proj = a_proj(a_emd_rev)
        if args.enable_domain_adv:

            a_emd_rev2 = grad_reverse(audio_emd, alpha=args.alpha_rev2)
            a_emd_proj2 = a_proj2(a_emd_rev2)


            a_domain_predict = domain_disc(a_emd_proj2)
            a_domain_loss = criterion(a_domain_predict, domain_labels)
            domain_adv_loss_local += a_domain_loss
            entropies['a'] = _entropy_from_logits(a_domain_predict)
        else:
            entropies['a'] = _entropy_from_logits(a_predict)


        a_modal_predict = modal_disc(a_emd_proj)
        a_modal_loss = criterion(a_modal_predict, audio_labels)
        modal_adv_loss += a_modal_loss


    T = 1.0
    active_modalities = [m for m in entropies.keys()]
    if active_modalities:
        total_exp = sum(torch.exp(entropies[m] / T) for m in active_modalities)
        for m in active_modalities:
            weights[m] = torch.exp(entropies[m] / T) / total_exp
    else:
        if args.use_video:
            weights['v'] = 1.0 / args.num_modals
        if args.use_flow:
            weights['f'] = 1.0 / args.num_modals
        if args.use_audio:
            weights['a'] = 1.0 / args.num_modals
    if args.use_video:
        cls_loss += weights.get('v', 1.0) * v_loss
    if args.use_flow:
        cls_loss += weights.get('f', 1.0) * f_loss
    if args.use_audio:
        cls_loss += weights.get('a', 1.0) * a_loss


    if args.use_video and args.use_flow and args.use_audio:
        feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
        if args.enable_domain_adv:
            feat_rev_global = torch.cat([v_emd_rev2, a_emd_rev2, f_emd_rev2], dim=1)

    elif args.use_video and args.use_flow:
        feat = torch.cat((v_emd, f_emd), dim=1)
        if args.enable_domain_adv:
            feat_rev_global = torch.cat([v_emd_rev2, f_emd_rev2], dim=1)

    elif args.use_video and args.use_audio:
        feat = torch.cat((v_emd, audio_emd), dim=1)
        if args.enable_domain_adv:
            feat_rev_global = torch.cat([v_emd_rev2, a_emd_rev2], dim=1)

    elif args.use_flow and args.use_audio:
        feat = torch.cat((f_emd, audio_emd), dim=1)
        if args.enable_domain_adv:
            feat_rev_global = torch.cat([f_emd_rev2, a_emd_rev2], dim=1)

    modal_adv_loss = modal_adv_loss / args.num_modals
    domain_adv_loss_local = domain_adv_loss_local / args.num_modals

    if args.enable_domain_adv:

        domain_pred_global = domain_disc_global(feat_rev_global)
        domain_adv_loss_global = criterion(domain_pred_global, domain_labels)
    else:
        domain_adv_loss_global = torch.tensor(0.0, device=feat.device)

    predict = mlp_cls(feat)
    loss = criterion(predict, labels)
    loss = (loss + args.domain_adv_loss_global * domain_adv_loss_global +
            args.domain_adv_loss_local * domain_adv_loss_local + args.modal_adv_loss * modal_adv_loss
            + args.cls_loss * cls_loss)


    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss


def validate_one_step(clip, labels, flow, spectrogram):
    labels = labels.cuda()
    with torch.no_grad():
        if args.use_video:
            clip = clip['imgs'].cuda().squeeze(1)
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            x_slow = x_slow + torch.randn_like(x_slow) * args.val_noise
            x_fast = x_fast + torch.randn_like(x_fast) * args.val_noise
            v_feat = (x_slow.detach(), x_fast.detach())
            v_feat = model.module.backbone.get_predict(v_feat)
            v_predict, v_emd = model.module.cls_head(v_feat)


        if args.use_audio:
            spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
            _, audio_feat, _ = audio_model(spectrogram)
            audio_feat = audio_feat + torch.randn_like(audio_feat) * args.val_noise
            a_predict, audio_emd = audio_cls_model(audio_feat.detach())


        if args.use_flow:
            flow = flow['imgs'].cuda().squeeze(1)
            f_feat = model_flow.module.backbone.get_feature(flow)
            f_feat = f_feat + torch.randn_like(f_feat) * args.val_noise
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

def test_one_step(clip, labels, flow, spectrogram):
    labels = labels.cuda()
    with torch.no_grad():
        if args.use_video:
            clip = clip['imgs'].cuda().squeeze(1)
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = (x_slow.detach(), x_fast.detach())
            v_feat = model.module.backbone.get_predict(v_feat)
            v_predict, v_emd = model.module.cls_head(v_feat)

        if args.use_audio:
            spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
            _, audio_feat, _ = audio_model(spectrogram)
            a_predict, audio_emd = audio_cls_model(audio_feat.detach())

        if args.use_flow:
            flow = flow['imgs'].cuda().squeeze(1)
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


class ProjectHead(nn.Module):
    def __init__(self, input_dim, out_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
    def forward(self, x):
        return self.proj(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, out_dim=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, out_dim)
        )
    def forward(self, x):
        return self.classifier(x)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, out_dim=2):
        super().__init__()
        self.domainclassifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.domainclassifier(x)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, choices=['epic', 'hac'])
    parser.add_argument('-s', '--source_domain', nargs='+', required=True, help='<Required> Set source_domain')
    parser.add_argument('-t', '--target_domain', nargs='+', required=True, help='<Required> Set target_domain')



    parser.add_argument('--datapath', type=str, default='/path/to/DATA_ROOT',
                        help='datapath')
    parser.add_argument('--num_class', type=int, required=True)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adamw', 'adam'],
                        help='optimizer: adamw, adam')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='lrweight_decay')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=15)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0,
                        help='BestAcc')
    parser.add_argument('--BestLoss', type=float, default=0,
                        help='BestLoss')
    parser.add_argument('--currentloss', type=float, default=0,
                        help='currentloss')
    parser.add_argument('--BestTestAcc', type=float, default=0,
                        help='BestTestAcc')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow',  action='store_true')
    parser.add_argument("--num_modals", type=int, default=2)

    parser.add_argument("--global_discriminator_hidden_dim", type=int, default=512)
    parser.add_argument("--project_out_dim", type=int, default=512)
    parser.add_argument("--alpha_rev", type=float, default=0.1)
    parser.add_argument("--alpha_rev2", type=float, default=0.3)
    parser.add_argument("--val_noise", type=float, default=0)

    parser.add_argument("--domain_adv_loss_global", type=float, default=0.5)
    parser.add_argument("--domain_adv_loss_local", type=float, default=0.5)
    parser.add_argument("--modal_adv_loss", type=float, default=0.1)
    parser.add_argument("--cls_loss", type=float, default=3.0)
    parser.add_argument('--num_workers', type=int, default=4)

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
    args.enable_domain_adv = (len(args.source_domain) > 1)
    args.domain_disc_out_dim = max(len(args.source_domain), 2)
    print(f'DG mode: {args.dg_mode}')
    if not args.enable_domain_adv:
        print('Single-source DG detected: disable domain adversarial losses in MDJA.')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    config_file = os.path.join(SCRIPT_DIR, 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py')
    checkpoint_file = os.path.join(SCRIPT_DIR, 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth')

    config_file_flow = os.path.join(SCRIPT_DIR, 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py')
    checkpoint_file_flow = os.path.join(SCRIPT_DIR, 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth')


    device = 'cuda:0'
    device = torch.device(device)


    input_dim = 0
    global_rev_input_dim = 0

    cfg = None
    cfg_flow = None

    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, args.num_class).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)


        input_dim = input_dim + 2304
        global_rev_input_dim = global_rev_input_dim + 2304

        v_proj = ProjectHead(input_dim=2304, out_dim=args.project_out_dim).cuda()
        v_proj2 = ProjectHead(input_dim=2304, out_dim=args.project_out_dim).cuda()



    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, args.num_class).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        input_dim = input_dim + 2048
        global_rev_input_dim = global_rev_input_dim + 2048

        f_proj = ProjectHead(input_dim=2048, out_dim=args.project_out_dim).cuda()
        f_proj2 = ProjectHead(input_dim=2048, out_dim=args.project_out_dim).cuda()



    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        checkpoint = torch.load(os.path.join(SCRIPT_DIR, "pretrained_models/vggsound_avgpool.pth.tar"))
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, args.num_class)
        audio_cls_model = audio_cls_model.cuda()

        input_dim = input_dim + 512
        global_rev_input_dim = global_rev_input_dim + 512

        a_proj = ProjectHead(input_dim=512, out_dim=args.project_out_dim).cuda()
        a_proj2 = ProjectHead(input_dim=512, out_dim=args.project_out_dim).cuda()

    mlp_cls = Encoder(input_dim=input_dim, out_dim=args.num_class)
    mlp_cls = mlp_cls.cuda()

    modal_disc = Classifier(input_dim=args.project_out_dim, out_dim=args.num_modals).cuda()
    domain_disc = DomainClassifier(input_dim=args.project_out_dim, out_dim=args.domain_disc_out_dim).cuda()


    domain_disc_global = Encoder(input_dim=global_rev_input_dim, out_dim=args.domain_disc_out_dim,
                                 hidden=args.global_discriminator_hidden_dim).cuda()


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


    params = list(mlp_cls.parameters()) + list(modal_disc.parameters()) + list(domain_disc.parameters()) + list(
        domain_disc_global.parameters())
    if args.use_video:
        params = (params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
            model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(
            v_proj.parameters()) + list(v_proj2.parameters()))

    if args.use_flow:
        params = (params + list(model_flow.module.backbone.layer4.parameters()) + list(
            model_flow.module.cls_head.parameters()) + list(f_proj.parameters()) + list(f_proj2.parameters()))
    if args.use_audio:
        params = (params + list(audio_cls_model.parameters()) + list(a_proj.parameters()) + list(a_proj2.parameters()))



    if args.optimizer == 'adamw':
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(params, lr=args.lr)

    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc

    currentloss = args.currentloss


    print("Training From Scratch ...")
    starting_epoch = 0
    print("starting_epoch: ", starting_epoch)

    train_dataset = DatasetClass(**make_dataset_kwargs(args, 'train', args.source_domain, cfg, cfg_flow, source=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=True, drop_last=True)
    val_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.source_domain, cfg, cfg_flow, source=True))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                 pin_memory=True, drop_last=False)


    if len(args.target_domain) == 1:
        test_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain, cfg, cfg_flow, source=False))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                                      shuffle=False, pin_memory=True, drop_last=False)
    else:
        test_dataset1 = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain[0:1], cfg, cfg_flow, source=False))
        test_dataset2 = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain[1:2], cfg, cfg_flow, source=False))
        test_dataloader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size, num_workers=args.num_workers,
                                                       shuffle=False, pin_memory=True, drop_last=False)
        test_dataloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size, num_workers=args.num_workers,
                                                       shuffle=False, pin_memory=True, drop_last=False)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:

        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("run_start,{}\n".format(datetime.now().isoformat()))
        f.write("dg_mode,{}\n".format(args.dg_mode))
        f.write("alpha_rev,{}\n".format(args.alpha_rev))
        f.write("alpha_rev2,{}\n".format(args.alpha_rev2))
        f.write("domain_adv_loss_global,{}\n".format(args.domain_adv_loss_global))
        f.write("domain_adv_loss_local,{}\n".format(args.domain_adv_loss_local))
        f.write("modal_adv_loss,{}\n".format(args.modal_adv_loss))
        f.write("cls_loss,{}\n".format(args.cls_loss))
        f.write("seed,{}\n".format(args.seed))
        f.write("\n")
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0


                print(split)
                mlp_cls.train(split == 'train')
                modal_disc.train(split == 'train')
                domain_disc.train(split == 'train')
                domain_disc_global.train(split == 'train')
                if args.use_video:
                    model.train(split == 'train')
                    v_proj.train(split == 'train')
                    v_proj2.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                    f_proj.train(split == 'train')
                    f_proj2.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
                    a_proj.train(split == 'train')
                    a_proj2.train(split == 'train')

                with tqdm.tqdm(total=len(dataloaders[split]), disable=True) as pbar:
                    for (i, (clip, flow, spectrogram, labels, domain_labels)) in enumerate(dataloaders[split]):
                        if split == 'train':
                            predict1, loss = train_one_step(clip, labels, flow, spectrogram, domain_labels)
                        else:
                            predict1, loss = validate_one_step(clip, labels, flow, spectrogram)


                        total_loss += loss.item() * batch_size
                        _, predict_gpu = torch.max(predict1.detach(), dim=1)
                        predict = predict_gpu.cpu()

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
                        currentloss = total_loss / float(count)
                        is_best = currentvalAcc >= BestAcc
                        if currentvalAcc >= BestAcc:
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)
                        if currentloss <= BestLoss:
                            BestLoss = total_loss / float(count)
                        if args.save_best and is_best:
                            torch.save(build_checkpoint(epoch_i), os.path.join(model_dir, log_name + '_best.pt'))
                        if args.save_checkpoint:
                            torch.save(build_checkpoint(epoch_i), os.path.join(model_dir, log_name + '.pt'))


                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    if split == 'val':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{} \n".format(BestEpoch,
                                                                                           BestLoss,
                                                                                           BestAcc))
                        f.flush()

            print('acc on epoch ', epoch_i)
            print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
            print('currentLoss', currentloss)
            print('BestLoss', BestLoss)
            print('BestValAcc ', BestAcc)
            print('BestTestAcc ', BestTestAcc)

        if len(args.target_domain) == 1:
            test_acc = 0
            test_count = 0
            test_total_loss = 0.0
            with tqdm.tqdm(total=len(test_dataloader), disable=True) as pbar:
                for i, (clip, flow, spectrogram, labels, domain_labels) in enumerate(test_dataloader):
                    predict1, loss = test_one_step(clip, labels, flow, spectrogram)
                    test_total_loss += loss.item() * batch_size
                    _, predict_gpu = torch.max(predict1.detach(), dim=1)
                    predict = predict_gpu.cpu()
                    test_acc += int((predict == labels).sum().item())
                    test_count += predict1.size()[0]
                    pbar.update()

            BestTestAcc = test_acc / float(test_count)
            f.write("{},test,{},{}\n".format(BestEpoch, test_total_loss / float(test_count), BestTestAcc))
        else:
            test_acc_1 = 0
            test_count_1 = 0
            test_total_loss_1 = 0.0
            with tqdm.tqdm(total=len(test_dataloader1), disable=True) as pbar:
                for i, (clip, flow, spectrogram, labels, domain_labels) in enumerate(test_dataloader1):
                    predict1, loss = test_one_step(clip, labels, flow, spectrogram)
                    test_total_loss_1 += loss.item() * batch_size
                    _, predict_gpu = torch.max(predict1.detach(), dim=1)
                    predict = predict_gpu.cpu()
                    test_acc_1 += int((predict == labels).sum().item())
                    test_count_1 += predict1.size()[0]
                    pbar.update()

            test_acc_2 = 0
            test_count_2 = 0
            test_total_loss_2 = 0.0
            with tqdm.tqdm(total=len(test_dataloader2), disable=True) as pbar:
                for i, (clip, flow, spectrogram, labels, domain_labels) in enumerate(test_dataloader2):
                    predict1, loss = test_one_step(clip, labels, flow, spectrogram)
                    test_total_loss_2 += loss.item() * batch_size
                    _, predict_gpu = torch.max(predict1.detach(), dim=1)
                    predict = predict_gpu.cpu()
                    test_acc_2 += int((predict == labels).sum().item())
                    test_count_2 += predict1.size()[0]
                    pbar.update()

            test_loss_1 = test_total_loss_1 / float(test_count_1)
            test_loss_2 = test_total_loss_2 / float(test_count_2)
            test_acc_1 = test_acc_1 / float(test_count_1)
            test_acc_2 = test_acc_2 / float(test_count_2)
            BestTestAcc = (test_acc_1 + test_acc_2) / 2.0

            f.write("{},test {},{},{}\n".format(BestEpoch, args.target_domain[0], test_loss_1, test_acc_1))
            f.write("{},test {},{},{}\n".format(BestEpoch, args.target_domain[1], test_loss_2, test_acc_2))
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
