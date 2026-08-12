#!/usr/bin/env bash
set -euo pipefail

# Fixed PatternMLP subnet validity experiments.
#
# Each run trains one budget-6 MLP pattern as a fixed subnet:
#   - all 12 attention branches are executed
#   - only the selected 6 MLP residual branches are executed
#   - no router decision is learned
#
# This isolates pattern quality before using the patterns inside a router bank.

DEFAULT_DATA_DIR="${HOME}/shared/hdd_ext/nvme1/public/vision/classification/imageNet"
DATA_DIR="${1:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
CUDA_IDS="${2:-${CUDA_VISIBLE_DEVICES:-2,8}}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-300}"
PATTERN_BANK="${PATTERN_BANK:-configs/pattern_banks/pairwise_low_risk_top4.yml}"
PATTERNS="${PATTERNS:-low_risk_02 low_risk_07 low_risk_06 low_risk_16}"

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_IDS}"
NPROC="${NPROC:-${#GPU_ARRAY[@]}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_IDS}"

echo "DATA_DIR=${DATA_DIR}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC=${NPROC}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "EPOCHS=${EPOCHS}"
echo "PATTERN_BANK=${PATTERN_BANK}"
echo "PATTERNS=${PATTERNS}"

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

for PATTERN in ${PATTERNS}; do
  echo "[Fixed Pattern] ${PATTERN}"
  torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
    --model pattern_mlp_vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --model-kwargs \
      pattern_mode=fixed \
      pattern_bank="${PATTERN_BANK}" \
      fixed_pattern="${PATTERN}" \
    "${COMMON_TRAIN_ARGS[@]}"
done

echo "Done. Check each run's args.yaml for fixed_pattern to identify results."
