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


METHOD_NAME = 'GMP'


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
        'domain_disc_state_dict': domain_disc.state_dict(),
    }
    if args.use_video:
        save['model_state_dict'] = model.state_dict()
        save['v_proj2_state_dict'] = v_proj2.state_dict()
    if args.use_flow:
        save['model_flow_state_dict'] = model_flow.state_dict()
        save['f_proj2_state_dict'] = f_proj2.state_dict()
    if args.use_audio:
        save['audio_model_state_dict'] = audio_model.state_dict()
        save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
        save['a_proj2_state_dict'] = a_proj2.state_dict()
    return save


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
    out = []
    for g in grads:
        out.append(None if g is None else (g * scale))
    return out


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

    modal_outputs = {}

    if args.use_video:
        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)
        v_cls_loss = criterion(v_predict, labels)
        if args.enable_domain_adv:
            v_domain_predict = domain_disc(v_proj2(grad_reverse(v_emd, alpha=args.alpha_rev2)))
            v_domain_loss = criterion(v_domain_predict, domain_labels)
        else:
            v_domain_predict = torch.zeros(v_predict.size(0), args.domain_disc_out_dim, device=v_predict.device)
            v_domain_loss = torch.tensor(0.0, device=v_predict.device)
        modal_outputs['v'] = {
            'emd': v_emd,
            'cls_logits': v_predict,
            'domain_logits': v_domain_predict,
            'cls_loss': v_cls_loss,
            'domain_loss': v_domain_loss,
            'params': video_modal_params,
        }

    if args.use_flow:
        f_feat = model_flow.module.backbone.get_predict(f_feat)
        f_predict, f_emd = model_flow.module.cls_head(f_feat)
        f_cls_loss = criterion(f_predict, labels)
        if args.enable_domain_adv:
            f_domain_predict = domain_disc(f_proj2(grad_reverse(f_emd, alpha=args.alpha_rev2)))
            f_domain_loss = criterion(f_domain_predict, domain_labels)
        else:
            f_domain_predict = torch.zeros(f_predict.size(0), args.domain_disc_out_dim, device=f_predict.device)
            f_domain_loss = torch.tensor(0.0, device=f_predict.device)
        modal_outputs['f'] = {
            'emd': f_emd,
            'cls_logits': f_predict,
            'domain_logits': f_domain_predict,
            'cls_loss': f_cls_loss,
            'domain_loss': f_domain_loss,
            'params': flow_modal_params,
        }

    if args.use_audio:
        a_predict, a_emd = audio_cls_model(audio_feat)
        a_cls_loss = criterion(a_predict, labels)
        if args.enable_domain_adv:
            a_domain_predict = domain_disc(a_proj2(grad_reverse(a_emd, alpha=args.alpha_rev2)))
            a_domain_loss = criterion(a_domain_predict, domain_labels)
        else:
            a_domain_predict = torch.zeros(a_predict.size(0), args.domain_disc_out_dim, device=a_predict.device)
            a_domain_loss = torch.tensor(0.0, device=a_predict.device)
        modal_outputs['a'] = {
            'emd': a_emd,
            'cls_logits': a_predict,
            'domain_logits': a_domain_predict,
            'cls_loss': a_cls_loss,
            'domain_loss': a_domain_loss,
            'params': audio_modal_params,
        }

    active_modals = list(modal_outputs.keys())
    active_count = max(len(active_modals), 1)

    sem_conf = {m: _mean_confidence(modal_outputs[m]['cls_logits'], labels) for m in active_modals}
    sem_ratio = _compute_diff_ratios(sem_conf)
    sem_coeff = {m: _mod_coeff_from_ratio_tanh(sem_ratio[m], args.alpha_k) for m in active_modals}
    if args.enable_domain_adv:
        dom_conf = {m: _mean_confidence(modal_outputs[m]['domain_logits'], domain_labels) for m in active_modals}
        dom_ratio = _compute_diff_ratios_domain(dom_conf)
        dom_coeff = {m: _mod_coeff_from_ratio_tanh(dom_ratio[m], args.alpha_p) for m in active_modals}
    else:
        dom_ratio = {m: torch.tensor(1.0, device=sem_conf[m].device) for m in active_modals}
        dom_coeff = {m: torch.tensor(1.0, device=sem_conf[m].device) for m in active_modals}

    per_modal_cls_grads = {}
    per_modal_dom_grads = {}
    final_modal_grads = {}

    for m in active_modals:
        params_m = modal_outputs[m]['params']
        g_cls = torch.autograd.grad(
            modal_outputs[m]['cls_loss'] * args.cls_loss,
            params_m,
            retain_graph=True,
            allow_unused=True,
        )
        if args.enable_domain_adv:
            g_dom = torch.autograd.grad(
                modal_outputs[m]['domain_loss'] * args.domain_adv_loss_local,
                params_m,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            g_dom = tuple(None for _ in params_m)

        g_cls = _scale_grads(g_cls, sem_coeff[m])
        g_dom = _scale_grads(g_dom, dom_coeff[m])

        if args.enable_domain_adv and _dot_grads(g_cls, g_dom) < 0:
            if sem_ratio[m] >= dom_ratio[m]:
                g_cls = _project_conflict(g_cls, g_dom)
            else:
                g_dom = _project_conflict(g_dom, g_cls)

        per_modal_cls_grads[m] = g_cls
        per_modal_dom_grads[m] = g_dom
        final_modal_grads[m] = _add_grads(g_cls, g_dom)

    emd_list = [modal_outputs[m]['emd'] for m in active_modals]
    feat = torch.cat(emd_list, dim=1)

    if args.enable_domain_adv:
        domain_adv_loss_local = sum(modal_outputs[m]['domain_loss'] for m in active_modals) / active_count
    else:
        domain_adv_loss_local = torch.tensor(0.0, device=feat.device)
    cls_loss = sum(modal_outputs[m]['cls_loss'] for m in active_modals) / active_count

    predict = mlp_cls(feat)
    loss = criterion(predict, labels)

    optim.zero_grad()

    loss.backward()

    for m in active_modals:
        params_m = modal_outputs[m]['params']
        grads_m = final_modal_grads[m]
        for p, g in zip(params_m, grads_m):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())

    optim.step()
    display_loss = (loss + args.domain_adv_loss_local * domain_adv_loss_local + args.cls_loss * cls_loss)
    return predict, display_loss


def validate_one_step(clip, labels, flow, spectrogram):
    labels = labels.cuda()
    with torch.no_grad():
        emd_list = []
        if args.use_video:
            clip = clip['imgs'].cuda().squeeze(1)
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            x_slow = x_slow + torch.randn_like(x_slow) * args.val_noise
            x_fast = x_fast + torch.randn_like(x_fast) * args.val_noise
            v_feat = (x_slow.detach(), x_fast.detach())
            v_feat = model.module.backbone.get_predict(v_feat)
            v_predict, v_emd = model.module.cls_head(v_feat)
            emd_list.append(v_emd)


        if args.use_audio:
            spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
            _, audio_feat, _ = audio_model(spectrogram)
            audio_feat = audio_feat + torch.randn_like(audio_feat) * args.val_noise
            a_predict, audio_emd = audio_cls_model(audio_feat.detach())
            emd_list.append(audio_emd)


        if args.use_flow:
            flow = flow['imgs'].cuda().squeeze(1)
            f_feat = model_flow.module.backbone.get_feature(flow)
            f_feat = f_feat + torch.randn_like(f_feat) * args.val_noise
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_predict, f_emd = model_flow.module.cls_head(f_feat)
            emd_list.append(f_emd)

        feat = torch.cat(emd_list, dim=1)

        predict = mlp_cls(feat)

    loss = criterion(predict, labels)

    return predict, loss

def test_one_step(clip, labels, flow, spectrogram):
    labels = labels.cuda()
    with torch.no_grad():
        emd_list = []
        if args.use_video:
            clip = clip['imgs'].cuda().squeeze(1)
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = (x_slow.detach(), x_fast.detach())
            v_feat = model.module.backbone.get_predict(v_feat)
            v_predict, v_emd = model.module.cls_head(v_feat)
            emd_list.append(v_emd)

        if args.use_audio:
            spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
            _, audio_feat, _ = audio_model(spectrogram)
            a_predict, audio_emd = audio_cls_model(audio_feat.detach())
            emd_list.append(audio_emd)

        if args.use_flow:
            flow = flow['imgs'].cuda().squeeze(1)
            f_feat = model_flow.module.backbone.get_feature(flow)
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_predict, f_emd = model_flow.module.cls_head(f_feat)
            emd_list.append(f_emd)

        feat = torch.cat(emd_list, dim=1)

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


def write_all_hparams(f, args):
    f.write("hparams_begin\n")
    for k in sorted(vars(args).keys()):
        f.write("{}={}\n".format(k, getattr(args, k)))
    f.write("hparams_end\n")
    f.write("\n")


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
    parser.add_argument('--use_flow', action='store_true')

    parser.add_argument("--num_modals", type=int, default=2)

    parser.add_argument("--project_out_dim", type=int, default=512)
    parser.add_argument("--alpha_rev2", type=float, default=0.3)
    parser.add_argument("--alpha_k", type=float, default=0.5,
                        help='suppress dominant semantic modality')
    parser.add_argument("--alpha_p", type=float, default=0.5,
                        help='suppress dominant domain-generalized modality')
    parser.add_argument("--val_noise", type=float, default=0)

    parser.add_argument("--domain_adv_loss_local", type=float, default=0.5)
    parser.add_argument("--cls_loss", type=float, default=3.0)
    parser.add_argument('--num_workers', type=int, default=8)

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
        print('Single-source DG detected: disable local domain adversarial loss in GMP.')

    if not (args.use_video or args.use_audio or args.use_flow):
        raise ValueError("At least one modality must be enabled via --use_video/--use_audio/--use_flow")

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
    cfg = None
    cfg_flow = None

    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, args.num_class).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)


        for p in model.module.parameters():
            p.requires_grad = False
        for p in model.module.backbone.fast_path.layer4.parameters():
            p.requires_grad = True
        for p in model.module.backbone.slow_path.layer4.parameters():
            p.requires_grad = True
        for p in model.module.cls_head.parameters():
            p.requires_grad = True


        input_dim = input_dim + 2304
        v_proj2 = ProjectHead(input_dim=2304, out_dim=args.project_out_dim).cuda()



    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, args.num_class).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)


        for p in model_flow.module.parameters():
            p.requires_grad = False
        for p in model_flow.module.backbone.layer4.parameters():
            p.requires_grad = True
        for p in model_flow.module.cls_head.parameters():
            p.requires_grad = True

        input_dim = input_dim + 2048
        f_proj2 = ProjectHead(input_dim=2048, out_dim=args.project_out_dim).cuda()



    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        checkpoint = torch.load(os.path.join(SCRIPT_DIR, "pretrained_models/vggsound_avgpool.pth.tar"))
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()
        for p in audio_model.parameters():
            p.requires_grad = False

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, args.num_class)
        audio_cls_model = audio_cls_model.cuda()

        input_dim = input_dim + 512
        a_proj2 = ProjectHead(input_dim=512, out_dim=args.project_out_dim).cuda()

    mlp_cls = Encoder(input_dim=input_dim, out_dim=args.num_class)
    mlp_cls = mlp_cls.cuda()

    domain_disc = DomainClassifier(input_dim=args.project_out_dim, out_dim=args.domain_disc_out_dim).cuda()


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


    params = list(mlp_cls.parameters()) + list(domain_disc.parameters())
    if args.use_video:
        params = (params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
            model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(
            v_proj2.parameters()))

    if args.use_flow:
        params = (params + list(model_flow.module.backbone.layer4.parameters()) + list(
            model_flow.module.cls_head.parameters()) + list(f_proj2.parameters()))
    if args.use_audio:
        params = (params + list(audio_cls_model.parameters()) + list(a_proj2.parameters()))

    video_modal_params = []
    flow_modal_params = []
    audio_modal_params = []
    if args.use_video:
        video_modal_params = (list(model.module.backbone.fast_path.layer4.parameters()) +
                              list(model.module.backbone.slow_path.layer4.parameters()) +
                              list(model.module.cls_head.parameters()))
    if args.use_flow:
        flow_modal_params = (list(model_flow.module.backbone.layer4.parameters()) +
                             list(model_flow.module.cls_head.parameters()))
    if args.use_audio:
        audio_modal_params = list(audio_cls_model.parameters())



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


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,
                                                   pin_memory=True, drop_last=True)
    val_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.source_domain, cfg, cfg_flow, source=True))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False,
                                                 pin_memory=True, drop_last=False)


    if len(args.target_domain) == 1:
        test_dataset = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain, cfg, cfg_flow, source=False))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8,
                                                      shuffle=False, pin_memory=True, drop_last=False)
    else:
        test_dataset1 = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain[0:1], cfg, cfg_flow, source=False))
        test_dataset2 = DatasetClass(**make_dataset_kwargs(args, 'test', args.target_domain[1:2], cfg, cfg_flow, source=False))
        test_dataloader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size, num_workers=8,
                                                       shuffle=False, pin_memory=True, drop_last=False)
        test_dataloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size, num_workers=8,
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
        write_all_hparams(f, args)
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0


                print(split)
                mlp_cls.train(split == 'train')
                domain_disc.train(split == 'train')
                if args.use_video:
                    model.train(split == 'train')
                    v_proj2.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                    f_proj2.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
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
