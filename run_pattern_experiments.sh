#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher for Q1-Q4 structured MLP subnet routing experiments.
#
# Usage:
#   bash run_pattern_experiments.sh [DATA_DIR] [GPU_IDS]
#
# Example:
#   bash run_pattern_experiments.sh ~/shared/hdd_ext/nvme1/public/vision/classification/imageNet 0,1
#
# Long training jobs are executed only when this script is run. The helper
# scripts create CSV manifests and launch train/analysis commands in sequence.

DEFAULT_DATA_DIR="${HOME}/shared/hdd_ext/nvme1/public/vision/classification/imageNet"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
CUDA_IDS="${2:-0,1}"
SEED="${SEED:-42}"
EPOCHS_FIXED="${EPOCHS_FIXED:-60}"
EPOCHS_ROUTER="${EPOCHS_ROUTER:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WORKERS="${WORKERS:-8}"
RESULTS_DIR="${RESULTS_DIR:-results}"
START_Q="${START_Q:-1}"

if [[ ! "${START_Q}" =~ ^[1-4]$ ]]; then
  echo "START_Q must be one of 1, 2, 3, or 4." >&2
  exit 2
fi

mkdir -p "${RESULTS_DIR}"

COMMON_TRAIN_ARGS=(
  --val-split val
  --aa rand-m9-mstd0.5-inc1
  --mixup .8
  --cutmix 1.0
  --aug-repeats 0
  --remode pixel
  --reprob 0.25
  --drop-path .1
  --opt adamw
  --weight-decay .05
  --sched cosine
  --lr 1e-3
  --warmup-lr 1e-6
  --min-lr 1e-5
  --warmup-epochs 5
  --smoothing 0.1
  --batch-size "${BATCH_SIZE}"
  --grad-accumulation 2
  -j "${WORKERS}"
  --amp
  --cuda "${CUDA_IDS}"
)

if (( START_Q <= 1 )); then
  echo "[Q1] Fixed 6-MLP depth-placement ablation"
  python scripts/run_fixed_pattern_ablation.py \
    --data-dir "${DATA_DIR}" \
    --bank configs/pattern_banks/patterns_6.yml \
    --patterns early6 mid6 late6 alternating6 \
    --epochs "${EPOCHS_FIXED}" \
    --batch-size "${BATCH_SIZE}" \
    --cuda "${CUDA_IDS}" \
    --seed "${SEED}" \
    --output-csv "${RESULTS_DIR}/fixed_pattern_ablation.csv" \
    --execute \
    "${COMMON_TRAIN_ARGS[@]}"
fi

if (( START_Q <= 2 )); then
  echo "[Q2-A] Shared-weight sampled supernet training for oracle analysis"
  torchrun --nproc_per_node=2 train_sh.py "${DATA_DIR}" \
    --model pattern_mlp_vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS_ROUTER}" \
    --seed "${SEED}" \
    --model-kwargs pattern_mode=sampled_supernet pattern_bank=configs/pattern_banks/patterns_6.yml fixed_pattern=early6 \
    "${COMMON_TRAIN_ARGS[@]}"

  echo "[Q2-B] Oracle analysis"
  if [[ -z "${ORACLE_CHECKPOINT:-}" ]]; then
    echo "Set ORACLE_CHECKPOINT=/path/to/model_best.pth.tar before running Q2-B on a trained sampled_supernet checkpoint."
  else
    python scripts/analyze_pattern_oracle.py \
      --data-dir "${DATA_DIR}" \
      --checkpoint "${ORACLE_CHECKPOINT}" \
      --bank configs/pattern_banks/patterns_6.yml \
      --batch-size 128 \
      --workers "${WORKERS}" \
      --output-csv "${RESULTS_DIR}/pattern_oracle_samples.csv" \
      --summary-csv "${RESULTS_DIR}/pattern_oracle_summary.csv"
  fi
fi

if (( START_Q <= 3 )); then
  echo "[Q3-A] Equal-cost router on 6-MLP bank"
  torchrun --nproc_per_node=2 train_sh.py "${DATA_DIR}" \
    --model pattern_mlp_vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS_ROUTER}" \
    --seed "${SEED}" \
    --model-kwargs pattern_mode=router pattern_bank=configs/pattern_banks/bank4.yml router_init=zero_logits \
    "${COMMON_TRAIN_ARGS[@]}"

  echo "[Q3-B] Mixed-cost router with budget targets"
  for TARGET in 6 8 10; do
    torchrun --nproc_per_node=2 train_sh.py "${DATA_DIR}" \
      --model pattern_mlp_vit_small_patch16_224_depth12 \
      --epochs "${EPOCHS_ROUTER}" \
      --seed "${SEED}" \
      --pattern-budget-target "${TARGET}" \
      --pattern-budget-weight "${BUDGET_WEIGHT:-0.01}" \
      --model-kwargs pattern_mode=router pattern_bank=configs/pattern_banks/patterns_mixed.yml router_init=zero_logits \
      "${COMMON_TRAIN_ARGS[@]}"
  done
fi

if (( START_Q <= 4 )); then
  echo "[Q4] Pattern bank size ablation"
  python scripts/run_pattern_bank_ablation.py \
    --data-dir "${DATA_DIR}" \
    --banks bank2 bank4 bank8 \
    --epochs "${EPOCHS_ROUTER}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --output-csv "${RESULTS_DIR}/pattern_bank_ablation.csv" \
    --execute \
    "${COMMON_TRAIN_ARGS[@]}"
fi

echo "Done. CSV manifests and summaries are under ${RESULTS_DIR}/."
