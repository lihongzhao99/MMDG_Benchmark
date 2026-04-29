import random
import os
import sys
from datetime import datetime
import torch
import argparse
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import MBCD_Model
from utils.dataloader import getdataloader
from utils.run_logging import append_run_log


parser = argparse.ArgumentParser(description="MMSA + MBCD domain generalization")

parser.add_argument(
    "--backbone", type=str, default="latefusion", help="latefusion/earlyfusion/mlp"
)
parser.add_argument(
    "--pretrained_model", type=str, default="", help="pretrained model path"
)
parser.add_argument(
    "--datapath",
    type=str,
    default="",
    help="dataset directory or .pkl file path",
)
parser.add_argument(
    "--stage",
    type=str,
    default="dg_train",
    help="dg_train/dg_test",
)
parser.add_argument(
    "--source_datasets",
    nargs="+",
    default=None,
    help="source domains for DG training, e.g. --source_datasets mosei mosi",
)
parser.add_argument(
    "--target_dataset",
    type=str,
    default="",
    help="target domain for frozen DG testing, e.g. --target_dataset sims",
)
parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
parser.add_argument(
    "--embed_dropout", type=float, default=0.25, help="embedding dropout"
)
parser.add_argument(
    "--res_dropout", type=float, default=0.1, help="residual block dropout"
)
parser.add_argument(
    "--out_dropout", type=float, default=0.0, help="output layer dropout"
)
parser.add_argument(
    "--nlevels",
    type=int,
    default=5,
    help="number of layers in the network (default: 5)",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=8,
    help="number of heads for the transformer network (default: 8)",
)
parser.add_argument("--proj_dim", type=int, default=40)
parser.add_argument(
    "--batch_size", type=int, default=24, metavar="N", help="batch size (default: 24)"
)
parser.add_argument(
    "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
)
parser.add_argument(
    "--lr", type=float, default=1e-4, help="initial learning rate (default: 1e-4)"
)
parser.add_argument(
    "--outer_lr",
    type=float,
    default=None,
    help="alias of --lr for outer-loop/update learning rate",
)
parser.add_argument(
    "--optim", type=str, default="Adam", help="optimizer to use (default: Adam)"
)
parser.add_argument(
    "--num_epochs", type=int, default=15, help="number of epochs (default: 15)"
)
parser.add_argument(
    "--when", type=int, default=30, help="when to decay learning rate (default: 30)"
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=730,
    help="frequency of result logging (default: 730)",
)
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
parser.add_argument("--gpu_id", type=int, default=0, help="single GPU id to use")
parser.add_argument(
    "--multi_gpu",
    action="store_true",
    help="enable DataParallel across all visible GPUs (default: disabled)",
)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--random_hparams_json", type=str, default="")

# ===== MBCD related arguments =====
parser.add_argument("--enable_mbcd", default=True, action="store_true", help="enable MBCD strategy")
parser.add_argument("--mbcd_hidden_dim", type=int, default=128)
parser.add_argument("--mbcd_modal_loss_weight", type=float, default=0.2)
parser.add_argument("--mbcd_fused_loss_weight", type=float, default=0.2)
parser.add_argument("--enable_layernorm", action="store_true", help="apply LayerNorm on modality features before MBCD heads")
parser.add_argument("--enable_grad_consistency", action="store_true", help="enable gradient consistency constraint between uni-modal and fused losses")
parser.add_argument("--gcc_weight", type=float, default=0.05, help="weight for gradient consistency term")
parser.add_argument("--gcc_alpha", type=float, default=0.1, help="alpha in Eq.(6)-style -alpha<g_uni,g_mm>")
parser.add_argument(
    "--inner_lr",
    type=float,
    default=None,
    help="inner-loop step size alias; mapped to --gcc_alpha",
)
parser.add_argument("--modality_drop_base", type=float, default=0.0)
parser.add_argument("--ema_beta", type=float, default=0.999)
parser.add_argument("--kl_mm_coeff", type=float, default=0.1)
parser.add_argument("--kl_um_coeff", type=float, default=0.05)

args = parser.parse_args()

# Backward-compatible aliases for meta-learning terminology.
if args.outer_lr is not None:
    args.lr = args.outer_lr
if args.inner_lr is not None:
    args.gcc_alpha = args.inner_lr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


use_cuda = torch.cuda.is_available() and (not args.no_cuda)
if use_cuda and (not args.multi_gpu):
    torch.cuda.set_device(args.gpu_id)

setup_seed(args.seed)

dataloder, orig_dim = getdataloader(args)
train_loader = dataloder["train"]
valid_loader = dataloder["valid"]
test_loader = dataloder["test"]
hyp_params = args
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
hyp_params.when = args.when
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = (
    len(train_loader),
    len(valid_loader),
    len(test_loader),
)
hyp_params.criterion = "L1Loss"
hyp_params.orig_dim = orig_dim
hyp_params.output_dim = 1

source_datasets = getattr(hyp_params, "source_datasets", None)
if source_datasets is None or len(source_datasets) == 0:
    source_datasets = [getattr(hyp_params, "dataset", "mosi")]
hyp_params.source_datasets = [str(ds).lower().strip() for ds in source_datasets]
if not getattr(hyp_params, "target_dataset", ""):
    hyp_params.target_dataset = getattr(hyp_params, "dataset", "")


if __name__ == "__main__":
    if hyp_params.stage in {"dg_train", "dg_test"}:
        run_started_at = datetime.now().isoformat()
        run_result = MBCD_Model.initiate(
            hyp_params, train_loader, valid_loader, test_loader
        )
        if not isinstance(run_result, dict):
            run_result = {"final_test_loss": float(run_result)}
        append_run_log(hyp_params, "mbcd", run_result, run_started_at=run_started_at)
    else:
        raise ValueError(
            f"Unsupported stage '{hyp_params.stage}'. Use dg_train or dg_test."
        )
