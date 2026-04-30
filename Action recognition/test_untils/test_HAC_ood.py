import argparse
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_DIR = os.path.dirname(SCRIPT_DIR)
if ACTION_DIR not in sys.path:
    sys.path.insert(0, ACTION_DIR)

from mmaction.apis import init_recognizer

import numpy as np
import torch
import torch.nn as nn
import tqdm

from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloaders.dataloader_HAC import HACDOMAIN
from dataloaders.dataloader_EPIC import EPICDOMAIN
import torch.nn.functional as F
from sklearn import metrics

# python test_HAC_ood.py  -s 'human' 'animal' -t 'cartoon' --use_video --use_audio --resumef models/log_ERM_human_animal2cartoon_video_audio_seed_0_best.pt 

to_np = lambda x: x.data.cpu().numpy()

def compute_all_metrics(conf, label):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)


    results = [fpr, auroc, aupr_in, aupr_out]

    return results

# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr



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
        # loss = criterion(predict, labels)

    return predict


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


def safe_load_state_dict(module, state_dict, name):
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError as e:
        print(f"[Warning] strict load failed for {name}: {e}")
        print(f"[Warning] Retry with strict=False for {name}")

    incompatible = module.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"[Warning] {name} missing_keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[Warning] {name} unexpected_keys (first 10): {unexpected[:10]}")


def normalize_hac_datapath(datapath: str) -> str:
    p = Path(datapath).expanduser().resolve()

    if p.name == "HAC" and p.parent.name == "HAC_Splits":
        base = p.parent.parent
        frame_root = p
    elif p.name == "HAC_Splits":
        base = p.parent
        frame_root = p / "HAC"
    else:
        base = p
        frame_root = base / "HAC"

    hac_link = base / "HAC"
    if frame_root.exists() and hac_link != frame_root:
        if not hac_link.exists():
            try:
                os.symlink(str(frame_root), str(hac_link), target_is_directory=True)
                print(f"[Info] Created symlink: {hac_link} -> {frame_root}")
            except OSError as e:
                print(f"[Warning] Failed to create symlink {hac_link} -> {frame_root}: {e}")
        elif hac_link.is_dir():
            try:
                is_empty = next(hac_link.iterdir(), None) is None
            except OSError:
                is_empty = False
            if is_empty:
                try:
                    hac_link.rmdir()
                    os.symlink(str(frame_root), str(hac_link), target_is_directory=True)
                    print(f"[Info] Replaced empty dir with symlink: {hac_link} -> {frame_root}")
                except OSError as e:
                    print(f"[Warning] Failed to replace {hac_link} with symlink: {e}")

    return str(base) + os.sep


def normalize_epic_datapath(datapath: str) -> str:
    """Normalize EPIC path to the MM-SADA split root expected by EPICDOMAIN."""
    p = Path(datapath).expanduser().resolve()
    split_files = [
        f"{domain}_{split}.pkl"
        for domain in ("D1", "D2", "D3")
        for split in ("train", "test")
    ]

    if p.name == "MM-SADA_Domain_Adaptation_Splits":
        return str(p) + os.sep
    if any((p / split_file).exists() for split_file in split_files):
        return str(p) + os.sep
    return str(p / "MM-SADA_Domain_Adaptation_Splits") + os.sep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source_domain", nargs="+", help="<Required> Set source_domain", required=True)
    parser.add_argument("-t", "--target_domain", nargs="+", help="<Required> Set target_domain", required=True)
    parser.add_argument(
        "--datapath",
        type=str,
        default="/path/to/HAC_DATA_ROOT",
        help="datapath",
    )
    parser.add_argument(
        "--datapath_epic",
        type=str,
        default="/path/to/EPIC_SPLIT_ROOT",
        help="datapath",
    )
    parser.add_argument("--bsz", type=int, default=8, help="batch_size")
    parser.add_argument("--appen", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_video", action="store_true")
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_flow", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument("--num_classes", type=int, default=7, help="num_classes")
    parser.add_argument(
        "--resumef",
        type=str,
        default="",
        help="Checkpoint path. If empty, use the default models/log_ERM..._best.pt path.",
    )
    args = parser.parse_args()
    args.datapath = normalize_hac_datapath(args.datapath)
    args.datapath_epic = normalize_epic_datapath(args.datapath_epic)

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
    num_classes = args.num_classes

    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, num_classes).to(device)
        cfg = model.cfg
        model = torch.nn.DataParallel(model).to(device)
        input_dim += 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, num_classes).to(device)
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
        audio_cls_model.fc = nn.Linear(512, num_classes)
        audio_cls_model = audio_cls_model.to(device)
        input_dim += 512

    mlp_cls = Encoder(input_dim=input_dim, out_dim=num_classes).to(device)
    resume_file = args.resumef if args.resumef else build_default_resume_path(args)
    if not os.path.exists(resume_file):
        raise FileNotFoundError(f"Checkpoint not found: {resume_file}")

    print("Loading checkpoint from", resume_file)
    checkpoint = torch.load(resume_file, map_location=device)

    if args.use_video:
        safe_load_state_dict(model, checkpoint["model_state_dict"], "video model")
    if args.use_flow:
        safe_load_state_dict(model_flow, checkpoint["model_flow_state_dict"], "flow model")
    if args.use_audio:
        safe_load_state_dict(audio_model, checkpoint["audio_model_state_dict"], "audio model")
        safe_load_state_dict(audio_cls_model, checkpoint["audio_cls_model_state_dict"], "audio cls model")
    safe_load_state_dict(mlp_cls, checkpoint["mlp_cls_state_dict"], "fusion classifier")

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

    test_dataset_ood = EPICDOMAIN(split='test', domain=['D1'], cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath_epic, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    test_dataloader_ood = torch.utils.data.DataLoader(test_dataset_ood, batch_size=args.bsz, num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=(device.type == "cuda"), drop_last=False)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    list_softmax = []
    list_label = []
    list_softmax_ood = []
    list_label_ood = []


    with tqdm.tqdm(total=len(test_dataloader)) as pbar:
        for clip, flow, spectrogram, labels in test_dataloader:
            predict = test_one_step(args, clip, labels, flow, spectrogram, criterion)
            batch_size = predict.size(0)

            pred_label = torch.argmax(predict.detach().cpu(), dim=1)
            total_correct += (pred_label == labels).sum().item()
            total_count += batch_size

            smax = to_np(F.softmax(predict, dim=1))
            list_softmax.extend(np.max(smax, axis=1)) 
            list_label.append(labels.cpu())

            pbar.update()

    for clip, flow, spectrogram, labels in test_dataloader_ood:
        predict = test_one_step(args, clip, labels, flow, spectrogram, criterion)

        smax = to_np(F.softmax(predict, dim=1))
        list_softmax_ood.extend(np.max(smax, axis=1)) 
        list_label_ood.append(labels.cpu())

    list_softmax = np.array(list_softmax)
    list_softmax_ood = np.array(list_softmax_ood)
    list_label = torch.cat(list_label).numpy().astype(int)
    list_label_ood = torch.cat(list_label_ood).numpy().astype(int)

    ood_gt = -1 * np.ones_like(list_label_ood)
    conf = np.concatenate([list_softmax, list_softmax_ood])
    label = np.concatenate([list_label, ood_gt])

    ood_metrics = compute_all_metrics(conf, label)

    print("Checkpoint epoch:", checkpoint.get("epoch", "N/A"))
    print("Saved BestValAcc:", checkpoint.get("BestAcc", "N/A"))
    print("Saved BestTestAcc:", checkpoint.get("BestTestAcc", "N/A"))
    print("Test Acc: {:.6f}".format(total_correct / float(total_count)))
    print("FPR@95: ", ood_metrics[0])
    print("AUROC: ", ood_metrics[1])
    print(
        "METRIC_OOD "
        "TestAcc={:.6f} FPR95={:.6f} AUROC={:.6f}".format(
            total_correct / float(total_count),
            float(ood_metrics[0]),
            float(ood_metrics[1]),
        )
    )
