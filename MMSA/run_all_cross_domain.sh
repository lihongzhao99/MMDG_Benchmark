#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run cross-domain multimodal sentiment-analysis experiments for one method.

Usage:
  ./run_all_cross_domain.sh --method METHOD [options] [-- extra training args]

Required:
  -m, --method METHOD       ERM, RNA, SimMMDG, MOOSA, CMRF, NEL, JAT, MBCD, GMP

Options:
  -s, --setting SETTING     multi, single, or all (default: all)
      --datapath PATH       Dataset directory or a concrete .pkl path
      --python PYTHON       Python executable (default: python)
      --dry-run             Print commands without running them
  -h, --help                Show this help

Examples:
  ./run_all_cross_domain.sh --method ERM --setting multi
  ./run_all_cross_domain.sh -m CMRF -s all --datapath ../data -- --num_epochs 5 --seed 1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
METHOD=""
SETTING="all"
DATAPATH=""
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--method)
      METHOD="$2"
      shift 2
      ;;
    -s|--setting)
      SETTING="${2,,}"
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
  erm) METHOD_NAME="ERM"; TRAIN_SCRIPT="train_MMSA_ERM.py" ;;
  rna|rna-net|rnanet) METHOD_NAME="RNA"; TRAIN_SCRIPT="train_MMSA_RNA.py" ;;
  simmmdg|sim-mmdg|sim_mmdg) METHOD_NAME="SimMMDG"; TRAIN_SCRIPT="train_MMSA_SimMMDG.py" ;;
  moosa) METHOD_NAME="MOOSA"; TRAIN_SCRIPT="train_MMSA_MOOSA.py" ;;
  cmrf) METHOD_NAME="CMRF"; TRAIN_SCRIPT="train_MMSA_CMRF.py" ;;
  nel) METHOD_NAME="NEL"; TRAIN_SCRIPT="train_MMSA_NEL.py" ;;
  jat|mdja) METHOD_NAME="JAT"; TRAIN_SCRIPT="train_MMSA_JAT.py" ;;
  mbcd) METHOD_NAME="MBCD"; TRAIN_SCRIPT="train_MMSA_MBCD.py" ;;
  gmp) METHOD_NAME="GMP"; TRAIN_SCRIPT="train_MMSA_GMP.py" ;;
  *)
    echo "Unsupported method: $METHOD" >&2
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

run_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run_one() {
  local sources="$1"
  local target="$2"
  read -r -a SOURCE_ARR <<< "$sources"
  local cmd=(
    "$PYTHON_BIN" "$TRAIN_SCRIPT"
    --source_datasets "${SOURCE_ARR[@]}"
    --target_dataset "$target"
  )
  if [[ -n "$DATAPATH" ]]; then
    cmd+=(--datapath "$DATAPATH")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  run_command "${cmd[@]}"
}

run_multi() {
  run_one "mosi mosei" "sims"
  run_one "mosi sims" "mosei"
}

run_single() {
  run_one "mosei" "sims"
  run_one "mosi" "sims"
  run_one "mosi" "mosei"
  run_one "sims" "mosi"
  run_one "sims" "mosei"
}

cd "$SCRIPT_DIR"
echo "Method: $METHOD_NAME"
echo "Setting: $SETTING"

if [[ "$SETTING" == "multi" || "$SETTING" == "all" ]]; then
  run_multi
fi

if [[ "$SETTING" == "single" || "$SETTING" == "all" ]]; then
  run_single
fi
