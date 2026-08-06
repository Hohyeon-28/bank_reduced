#!/usr/bin/env bash
set -euo pipefail

# Joint random-bank PatternMLP ViT experiment.
#
# This script trains the ViT weights and the pattern router together from
# scratch. During training, a small fraction of batches use quota-balanced
# random pattern assignment so every pattern keeps receiving gradients.

DEFAULT_DATA_DIR="${HOME}/shared/hdd_ext/nvme1/public/vision/classification/imageNet"
DATA_DIR="${1:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
CUDA_IDS="${2:-${CUDA_VISIBLE_DEVICES:-2,8}}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-8}"
EPOCHS_ROUTER="${EPOCHS_ROUTER:-300}"
EPOCHS_BASELINE="${EPOCHS_BASELINE:-300}"
TARGET_BUDGET="${TARGET_BUDGET:-6}"
BUDGET_WEIGHT="${BUDGET_WEIGHT:-0.01}"
PATTERN_BANK="${PATTERN_BANK:-configs/pattern_banks/random_budget6_bank12_seed42.yml}"
RUN_BASELINE="${RUN_BASELINE:-0}"

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_IDS}"
NPROC="${NPROC:-${#GPU_ARRAY[@]}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_IDS}"

echo "DATA_DIR=${DATA_DIR}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC=${NPROC}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "PATTERN_BANK=${PATTERN_BANK}"
echo "TARGET_BUDGET=${TARGET_BUDGET}"

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

echo "[Joint] Router + ViT training with random pattern exploration"
torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
  --model pattern_mlp_vit_small_patch16_224_depth12 \
  --epochs "${EPOCHS_ROUTER}" \
  --seed "${SEED}" \
  --pattern-budget-target "${TARGET_BUDGET}" \
  --pattern-budget-weight "${BUDGET_WEIGHT}" \
  --model-kwargs \
    pattern_mode=router \
    pattern_bank="${PATTERN_BANK}" \
    router_init=zero_logits \
    router_random_start_prob=0.5 \
    router_random_mid_prob=0.2 \
    router_random_end_prob=0.05 \
  "${COMMON_TRAIN_ARGS[@]}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "[Baseline] ViT-S baseline training"
  torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
    --model vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS_BASELINE}" \
    --seed "${SEED}" \
    "${COMMON_TRAIN_ARGS[@]}"
fi

echo "Done. Use scripts/analyze_pattern_results.py to summarize output/train into results/analysis."
