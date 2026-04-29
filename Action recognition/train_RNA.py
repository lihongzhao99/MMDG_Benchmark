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


METHOD_NAME = 'RNA'


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

    if args.use_flow:
        f_feat = model_flow.module.backbone.get_predict(f_feat.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

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

    predict = mlp_cls(feat)
    loss = criterion(predict, labels)

    # RNA loss
    if args.use_video and args.use_flow and args.use_audio:
        video_norm = v_emd.norm(p=2, dim=1).mean()
        flow_norm = f_emd.norm(p=2, dim=1).mean()
        audio_norm = audio_emd.norm(p=2, dim=1).mean()
        feat_frac1 = video_norm / audio_norm
        feat_frac2 = video_norm / flow_norm
        feat_frac3 = flow_norm / audio_norm
        loss_RNA = ((feat_frac1 - 1) ** 2 + (feat_frac2 - 1) ** 2 + (feat_frac3 - 1) ** 2) / 3
    elif args.use_video and args.use_flow:
        video_norm = v_emd.norm(p=2, dim=1).mean()
        flow_norm = f_emd.norm(p=2, dim=1).mean()
        feat_frac = video_norm / flow_norm
        loss_RNA = (feat_frac - 1) ** 2
    elif args.use_video and args.use_audio:
        video_norm = v_emd.norm(p=2, dim=1).mean()
        audio_norm = audio_emd.norm(p=2, dim=1).mean()
        feat_frac = video_norm / audio_norm
        loss_RNA = (feat_frac - 1) ** 2
    elif args.use_flow and args.use_audio:
        flow_norm = f_emd.norm(p=2, dim=1).mean()
        audio_norm = audio_emd.norm(p=2, dim=1).mean()
        feat_frac = flow_norm / audio_norm
        loss_RNA = (feat_frac - 1) ** 2

    loss = loss + args.alpha_RNA*loss_RNA

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
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0,
                        help='BestAcc')
    parser.add_argument('--BestTestAcc', type=float, default=0,
                        help='BestTestAcc')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--alpha_RNA', type=float, default=1.0,
                        help='alpha_RNA')
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow', action='store_true')
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

        input_dim = input_dim + 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, num_classes).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

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

        input_dim = input_dim + 512

    mlp_cls = Encoder(input_dim=input_dim, out_dim=num_classes)
    mlp_cls = mlp_cls.cuda()


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

    params = list(mlp_cls.parameters())
    if args.use_video:
        params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) 
    if args.use_flow:
        params = params + list(model_flow.module.backbone.layer4.parameters()) +list(model_flow.module.cls_head.parameters()) 
    if args.use_audio:
        params = params + list(audio_cls_model.parameters())

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
        if args.use_flow:
            model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
        if args.use_audio:
            audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
            audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
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
                if args.use_flow:
                    model_flow.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
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
                                    save['model_state_dict'] = model.state_dict()
                                if args.use_flow:
                                    save['model_flow_state_dict'] = model_flow.state_dict()
                                if args.use_audio:
                                    save['audio_model_state_dict'] = audio_model.state_dict()
                                    save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()

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
                                save['model_state_dict'] = model.state_dict()
                            if args.use_flow:
                                save['model_flow_state_dict'] = model_flow.state_dict()
                            if args.use_audio:
                                save['audio_model_state_dict'] = audio_model.state_dict()
                                save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()

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
