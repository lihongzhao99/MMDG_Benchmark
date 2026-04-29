#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run cross-domain action-recognition experiments for one method.

Usage:
  ./run_all_cross_domain.sh --method METHOD [options] [-- extra training args]

Required:
  -m, --method METHOD        ERM, RNA, SimMMDG, MOOSA, CMRF, NEL, JAT, MBCD, GMP

Options:
  -d, --dataset DATASET      epic, hac, or all (default: all)
  -s, --setting SETTING      multi, single, or all (default: all)
  -M, --modality MODALITY    va, vf, af, vaf, or all (default: all)
      --datapath PATH        Dataset root passed to the training script
      --python PYTHON        Python executable (default: python)
      --dry-run              Print commands without running them
  -h, --help                 Show this help

Examples:
  ./run_all_cross_domain.sh --method ERM --dataset epic --setting multi --modality va
  ./run_all_cross_domain.sh -m MBCD -d all -s all -M all -- --nepochs 5 --seed 1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
METHOD=""
DATASET="all"
SETTING="all"
MODALITY="all"
DATAPATH=""
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--method)
      METHOD="$2"
      shift 2
      ;;
    -d|--dataset)
      DATASET="${2,,}"
      shift 2
      ;;
    -s|--setting)
      SETTING="${2,,}"
      shift 2
      ;;
    -M|--modality)
      MODALITY="$(printf '%s' "${2,,}" | tr -d '+_-')"
      shift 2
      ;;
    --datapath)
      DATAPATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$METHOD" ]]; then
  echo "Missing required --method." >&2
  usage >&2
  exit 1
fi

case "${METHOD,,}" in
  erm) METHOD_NAME="ERM"; TRAIN_SCRIPT="train_ERM.py" ;;
  rna|rna-net|rnanet) METHOD_NAME="RNA"; TRAIN_SCRIPT="train_RNA.py" ;;
  simmmdg|sim-mmdg|sim_mmdg) METHOD_NAME="SimMMDG"; TRAIN_SCRIPT="train_SimMMDG.py" ;;
  moosa) METHOD_NAME="MOOSA"; TRAIN_SCRIPT="train_MOOSA.py" ;;
  cmrf) METHOD_NAME="CMRF"; TRAIN_SCRIPT="train_CMRF.py" ;;
  nel) METHOD_NAME="NEL"; TRAIN_SCRIPT="train_NEL.py" ;;
  jat) METHOD_NAME="JAT"; TRAIN_SCRIPT="train_JAT.py" ;;
  mbcd) METHOD_NAME="MBCD"; TRAIN_SCRIPT="train_MBCD.py" ;;
  gmp) METHOD_NAME="GMP"; TRAIN_SCRIPT="train_GMP.py" ;;
  *)
    echo "Unsupported method: $METHOD" >&2
    usage >&2
    exit 1
    ;;
esac

case "$DATASET" in
  epic|hac|all) ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    usage >&2
    exit 1
    ;;
esac

case "$SETTING" in
  multi|single|all) ;;
  *)
    echo "Unsupported setting: $SETTING" >&2
    usage >&2
    exit 1
    ;;
esac

case "$MODALITY" in
  va|vf|af|vaf|all) ;;
  *)
    echo "Unsupported modality: $MODALITY" >&2
    usage >&2
    exit 1
    ;;
esac

run_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

modality_flags() {
  local modal="$1"
  MODAL_FLAGS=()
  NUM_MODALS=2
  case "$modal" in
    va) MODAL_FLAGS=(--use_video --use_audio); NUM_MODALS=2 ;;
    vf) MODAL_FLAGS=(--use_video --use_flow); NUM_MODALS=2 ;;
    af) MODAL_FLAGS=(--use_audio --use_flow); NUM_MODALS=2 ;;
    vaf) MODAL_FLAGS=(--use_video --use_audio --use_flow); NUM_MODALS=3 ;;
  esac
}

run_one() {
  local dataset="$1"
  local sources="$2"
  local target="$3"
  local modal="$4"
  local num_class

  if [[ "$dataset" == "epic" ]]; then
    num_class=8
  else
    num_class=7
  fi

  modality_flags "$modal"
  read -r -a SOURCE_ARR <<< "$sources"
  read -r -a TARGET_ARR <<< "$target"

  local cmd=(
    "$PYTHON_BIN" "$TRAIN_SCRIPT"
    --dataset "$dataset"
    --num_class "$num_class"
    -s "${SOURCE_ARR[@]}"
    -t "${TARGET_ARR[@]}"
    "${MODAL_FLAGS[@]}"
  )

  if [[ "$METHOD_NAME" == "JAT" || "$METHOD_NAME" == "GMP" ]]; then
    cmd+=(--num_modals "$NUM_MODALS")
  fi

  if [[ -n "$DATAPATH" ]]; then
    cmd+=(--datapath "$DATAPATH")
  fi

  cmd+=("${EXTRA_ARGS[@]}")
  run_command "${cmd[@]}"
}

run_epic_multi() {
  local modal="$1"
  run_one epic "D2 D3" "D1" "$modal"
  run_one epic "D1 D3" "D2" "$modal"
  run_one epic "D1 D2" "D3" "$modal"
}

run_epic_single() {
  local modal="$1"
  run_one epic "D1" "D2" "$modal"
  run_one epic "D1" "D3" "$modal"
  run_one epic "D2" "D1" "$modal"
  run_one epic "D2" "D3" "$modal"
  run_one epic "D3" "D1" "$modal"
  run_one epic "D3" "D2" "$modal"
}

run_hac_multi() {
  local modal="$1"
  run_one hac "animal cartoon" "human" "$modal"
  run_one hac "human cartoon" "animal" "$modal"
  run_one hac "human animal" "cartoon" "$modal"
}

run_hac_single() {
  local modal="$1"
  run_one hac "human" "animal" "$modal"
  run_one hac "human" "cartoon" "$modal"
  run_one hac "animal" "human" "$modal"
  run_one hac "animal" "cartoon" "$modal"
  run_one hac "cartoon" "human" "$modal"
  run_one hac "cartoon" "animal" "$modal"
}

modalities=()
if [[ "$MODALITY" == "all" ]]; then
  modalities=(va vf af vaf)
else
  modalities=("$MODALITY")
fi

cd "$SCRIPT_DIR"
echo "Method: $METHOD_NAME"
echo "Dataset: $DATASET | Setting: $SETTING | Modality: ${modalities[*]}"

for modal in "${modalities[@]}"; do
  if [[ "$DATASET" == "epic" || "$DATASET" == "all" ]]; then
    if [[ "$SETTING" == "multi" || "$SETTING" == "all" ]]; then
      run_epic_multi "$modal"
    fi
    if [[ "$SETTING" == "single" || "$SETTING" == "all" ]]; then
      run_epic_single "$modal"
    fi
  fi

  if [[ "$DATASET" == "hac" || "$DATASET" == "all" ]]; then
    if [[ "$SETTING" == "multi" || "$SETTING" == "all" ]]; then
      run_hac_multi "$modal"
    fi
    if [[ "$SETTING" == "single" || "$SETTING" == "all" ]]; then
      run_hac_single "$modal"
    fi
  fi
done
