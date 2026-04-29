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
import torch.nn.functional as F
from sklearn import metrics

# python test_HAC_misd.py  -s 'human' 'animal' -t 'cartoon' --use_video --use_audio --resumef models/log_ERM_human_animal2cartoon_video_audio_seed_0_best.pt 


to_np = lambda x: x.data.cpu().numpy()

def calc_aurc_eaurc(softmax_max, correct):
    # softmax = np.array(softmax)
    correctness = np.array(correct)
    # softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x: x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def calc_fpr_aupr(softmax_max, correct):
    # softmax = np.array(softmax)
    correctness = np.array(correct)
    # softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    tnr_in_tpr_95 = 1 - fpr[np.argmax(tpr >= .95)]

    precision, recall, thresholds = metrics.precision_recall_curve(correctness, softmax_max)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95, tnr_in_tpr_95


def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list


# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area
    return aurc, eaurc


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
    """Normalize user HAC path to dataloader-compatible base path.

    dataloader_HAC expects a base path where:
      - CSV/audio/video live under <base>/HAC_Splits/
      - temp/frame dir is <base>/HAC/

        This function accepts user-friendly inputs like:
            - /.../DATA_ROOT/HAC_Splits/HAC
            - /.../DATA_ROOT/HAC_Splits
            - /.../DATA_ROOT
        and returns a valid <base> path string.
    """
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

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    list_softmax = []
    list_correct = []

    with tqdm.tqdm(total=len(test_dataloader)) as pbar:
        for clip, flow, spectrogram, labels in test_dataloader:
            predict, loss = test_one_step(args, clip, labels, flow, spectrogram, criterion)
            batch_size = predict.size(0)

            total_loss += loss.item() * batch_size
            pred_label = torch.argmax(predict.detach().cpu(), dim=1)
            total_correct += (pred_label == labels).sum().item()
            total_count += batch_size

            smax = to_np(F.softmax(predict, dim=1))
            list_softmax.extend(np.max(smax, axis=1)) 

            for j in range(len(pred_label)):
                if pred_label[j] == labels[j]:
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)
    
            pbar.set_postfix_str(
                "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(
                    total_loss / float(total_count),
                    loss.item(),
                    total_correct / float(total_count),
                )
            )
            pbar.update()

    list_softmax = np.array(list_softmax)
    aurc, eaurc = calc_aurc_eaurc(list_softmax, list_correct)
    auroc, aupr_success, aupr, fpr, tnr = calc_fpr_aupr(list_softmax, list_correct)

    print("Checkpoint epoch:", checkpoint.get("epoch", "N/A"))
    print("Saved BestValAcc:", checkpoint.get("BestAcc", "N/A"))
    print("Saved BestTestAcc:", checkpoint.get("BestTestAcc", "N/A"))
    print("Test Loss: {:.6f}".format(total_loss / float(total_count)))
    print("Test Acc: {:.6f}".format(total_correct / float(total_count)))
    print("AURC {0:.2f}".format(aurc * 1000))
    print("AUROC {0:.2f}".format(auroc * 100))
    print('FPR95 {0:.2f}'.format(fpr * 100))
    print(
        "METRIC_MISD "
        "TestAcc={:.6f} AURC={:.2f} AUROC={:.2f} FPR95={:.2f}".format(
            total_correct / float(total_count),
            aurc * 1000,
            auroc * 100,
            fpr * 100,
        )
    )
