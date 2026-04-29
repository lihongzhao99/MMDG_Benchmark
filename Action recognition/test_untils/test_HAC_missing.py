from mmaction.apis import init_recognizer
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm

from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloaders.dataloader_HAC import HACDOMAIN

# python test_HAC_missing.py  -s 'human' 'animal' -t 'cartoon' --use_video --use_audio --zero_audio --resumef models/log_ERM_human_animal2cartoon_video_audio_seed_0_best.pt --datapath /path/to/HAC_DATA_ROOT

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


def build_fusion_feature(args, embeddings):
    feat_list = []
    for name in ["video", "audio", "flow"]:
        if not getattr(args, f"use_{name}"):
            continue

        emd = embeddings[name]
        if getattr(args, f"zero_{name}"):
            emd = torch.zeros_like(emd)
        feat_list.append(emd)

    if not feat_list:
        raise ValueError("At least one modality must be enabled.")

    if len(feat_list) == 1:
        return feat_list[0]
    return torch.cat(feat_list, dim=1)


def extract_embeddings(args, clip, flow, spectrogram):
    embeddings = {}

    with torch.no_grad():
        if args.use_video:
            clip = clip["imgs"].to(device, non_blocking=True).squeeze(1)
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = model.module.backbone.get_predict((x_slow.detach(), x_fast.detach()))
            _, embeddings["video"] = model.module.cls_head(v_feat)

        if args.use_flow:
            flow = flow["imgs"].to(device, non_blocking=True).squeeze(1)
            f_feat = model_flow.module.backbone.get_feature(flow)
            f_feat = model_flow.module.backbone.get_predict(f_feat.detach())
            _, embeddings["flow"] = model_flow.module.cls_head(f_feat)

        if args.use_audio:
            spectrogram = spectrogram.unsqueeze(1).float().to(device, non_blocking=True)
            _, audio_feat, _ = audio_model(spectrogram)
            _, embeddings["audio"] = audio_cls_model(audio_feat.detach())

    return embeddings


def test_one_step(args, clip, labels, flow, spectrogram, criterion):
    labels = labels.to(device, non_blocking=True)
    embeddings = extract_embeddings(args, clip, flow, spectrogram)
    feat = build_fusion_feature(args, embeddings)

    with torch.no_grad():
        predict = mlp_cls(feat)
        loss = criterion(predict, labels)

    return predict, loss


def build_default_resume_path(args):
    log_name = "log_ERM%s2%s" % (args.source_domain, args.target_domain)
    if args.use_video:
        log_name = log_name + "_video"
    if args.use_flow:
        log_name = log_name + "_flow"
    if args.use_audio:
        log_name = log_name + "_audio"
    log_name = log_name + "_seed_" + str(args.seed)
    log_name = log_name + args.appen
    return os.path.join("models", log_name + "_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source_domain", nargs="+", help="<Required> Set source_domain", required=True)
    parser.add_argument("-t", "--target_domain", nargs="+", help="<Required> Set target_domain", required=True)
    parser.add_argument(
        "--datapath",
        type=str,
        default="/cluster/work/ibk_chatzi/hao/ActorShift/",
        help="datapath",
    )
    parser.add_argument("--bsz", type=int, default=8, help="batch_size")
    parser.add_argument("--appen", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_video", action="store_true")
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_flow", action="store_true")
    parser.add_argument("--zero_video", action="store_true")
    parser.add_argument("--zero_audio", action="store_true")
    parser.add_argument("--zero_flow", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2, help="num_workers")
    parser.add_argument(
        "--resumef",
        type=str,
        default="",
        help="Checkpoint path. If empty, use the default models/log_ERM..._best.pt path.",
    )
    args = parser.parse_args()

    if not (args.use_video or args.use_flow or args.use_audio):
        raise ValueError("Please enable at least one modality with --use_video, --use_flow, or --use_audio.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_file = "configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py"
    checkpoint_file = "pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth"

    config_file_flow = "configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py"
    checkpoint_file_flow = "pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_dim = 0
    cfg = None
    cfg_flow = None

    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, 7).to(device)
        cfg = model.cfg
        model = torch.nn.DataParallel(model).to(device)
        input_dim += 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, 7).to(device)
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow).to(device)
        input_dim += 2048

    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        audio_checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar", map_location=device)
        audio_model.load_state_dict(audio_checkpoint["model_state_dict"])
        audio_model = audio_model.to(device)

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(audio_checkpoint["model_state_dict"], strict=False)
        audio_cls_model.fc = nn.Linear(512, 7)
        audio_cls_model = audio_cls_model.to(device)
        input_dim += 512

    mlp_cls = Encoder(input_dim=input_dim, out_dim=7).to(device)

    resume_file = args.resumef if args.resumef else build_default_resume_path(args)
    if not os.path.exists(resume_file):
        raise FileNotFoundError(f"Checkpoint not found: {resume_file}")

    print("Loading checkpoint from", resume_file)
    checkpoint = torch.load(resume_file, map_location=device)

    if args.use_video:
        model.load_state_dict(checkpoint["model_state_dict"])
    if args.use_flow:
        model_flow.load_state_dict(checkpoint["model_flow_state_dict"])
    if args.use_audio:
        audio_model.load_state_dict(checkpoint["audio_model_state_dict"])
        audio_cls_model.load_state_dict(checkpoint["audio_cls_model_state_dict"])
    mlp_cls.load_state_dict(checkpoint["mlp_cls_state_dict"])

    if args.use_video:
        model.eval()
    if args.use_flow:
        model_flow.eval()
    if args.use_audio:
        audio_model.eval()
        audio_cls_model.eval()
    mlp_cls.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    test_dataset = HACDOMAIN(
        split="test",
        source=False,
        domain=args.target_domain,
        cfg=cfg,
        cfg_flow=cfg_flow,
        datapath=args.datapath,
        use_video=args.use_video,
        use_flow=args.use_flow,
        use_audio=args.use_audio,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.bsz,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with tqdm.tqdm(total=len(test_dataloader)) as pbar:
        for clip, flow, spectrogram, labels in test_dataloader:
            predict, loss = test_one_step(args, clip, labels, flow, spectrogram, criterion)
            batch_size = predict.size(0)

            total_loss += loss.item() * batch_size
            pred_label = torch.argmax(predict.detach().cpu(), dim=1)
            total_correct += (pred_label == labels).sum().item()
            total_count += batch_size

            pbar.set_postfix_str(
                "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(
                    total_loss / float(total_count),
                    loss.item(),
                    total_correct / float(total_count),
                )
            )
            pbar.update()

    print("Checkpoint epoch:", checkpoint.get("epoch", "N/A"))
    print("Saved BestValAcc:", checkpoint.get("BestAcc", "N/A"))
    print("Saved BestTestAcc:", checkpoint.get("BestTestAcc", "N/A"))
    print("Zeroed modalities: video={}, flow={}, audio={}".format(
        args.zero_video, args.zero_flow, args.zero_audio
    ))
    print("Test Loss: {:.6f}".format(total_loss / float(total_count)))
    print("Test Acc: {:.6f}".format(total_correct / float(total_count)))
