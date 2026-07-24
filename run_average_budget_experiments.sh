#!/usr/bin/env bash
set -euo pipefail

# Average-budget adaptive MLP routing experiment.
#
# Motivation:
#   Train candidate patterns fairly with uniform sampling, then train a router
#   that keeps the average active MLP count near the target budget.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash run_average_budget_experiments.sh [DATA_DIR] [GPU_IDS]
#
# Common overrides:
#   EPOCHS_SUPERNET=300 EPOCHS_ROUTER=100 TARGET_BUDGETS="6" bash run_average_budget_experiments.sh
#   STAGE=router SUPERNET_CHECKPOINT=/path/to/model_best.pth.tar bash run_average_budget_experiments.sh

DEFAULT_DATA_DIR="${HOME}/shared/hdd_ext/nvme1/public/vision/classification/imageNet"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
CUDA_IDS="${2:-${CUDA_VISIBLE_DEVICES:-0,1}}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WORKERS="${WORKERS:-8}"
EPOCHS_SUPERNET="${EPOCHS_SUPERNET:-300}"
EPOCHS_ROUTER="${EPOCHS_ROUTER:-100}"
TARGET_BUDGETS="${TARGET_BUDGETS:-6}"
BUDGET_WEIGHT="${BUDGET_WEIGHT:-0.01}"
PATTERN_BANK="${PATTERN_BANK:-configs/pattern_banks/mixed_budget_v1.yml}"
RESULTS_DIR="${RESULTS_DIR:-results/average_budget}"
STAGE="${STAGE:-all}"
FREEZE_ROUTER_BACKBONE="${FREEZE_ROUTER_BACKBONE:-1}"

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_IDS}"
NPROC="${NPROC:-${#GPU_ARRAY[@]}}"

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

find_latest_checkpoint() {
  local latest_dir
  latest_dir=$(find output/train -maxdepth 1 -type d -name 'pattern_mlp_vit_small_patch16_224_depth12_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')
  if [[ -z "${latest_dir}" ]]; then
    return 1
  fi
  if [[ -f "${latest_dir}/model_best.pth.tar" ]]; then
    printf '%s\n' "${latest_dir}/model_best.pth.tar"
  else
    find "${latest_dir}" -maxdepth 1 -type f -name 'checkpoint-*.pth.tar' | sort -V | tail -n 1
  fi
}

if [[ "${STAGE}" == "all" || "${STAGE}" == "supernet" ]]; then
  echo "[Stage A] Uniform sampled supernet training"
  torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
    --model pattern_mlp_vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS_SUPERNET}" \
    --seed "${SEED}" \
    --model-kwargs pattern_mode=sampled_uniform pattern_bank="${PATTERN_BANK}" fixed_pattern=late6 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "router" ]]; then
  if [[ -z "${SUPERNET_CHECKPOINT:-}" ]]; then
    SUPERNET_CHECKPOINT="$(find_latest_checkpoint || true)"
  fi
  if [[ -z "${SUPERNET_CHECKPOINT}" ]]; then
    echo "SUPERNET_CHECKPOINT is required for router stage, or run STAGE=all after supernet training." >&2
    exit 2
  fi

  echo "Using supernet checkpoint: ${SUPERNET_CHECKPOINT}"
  printf '%s\n' "${SUPERNET_CHECKPOINT}" > "${RESULTS_DIR}/supernet_checkpoint.txt"

  ROUTER_FREEZE_ARGS=()
  if [[ "${FREEZE_ROUTER_BACKBONE}" == "1" ]]; then
    ROUTER_FREEZE_ARGS=(--freeze-backbone-for-router)
  fi

  for TARGET in ${TARGET_BUDGETS}; do
    echo "[Stage C] Router training with average budget target ${TARGET}"
    torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
      --model pattern_mlp_vit_small_patch16_224_depth12 \
      --epochs "${EPOCHS_ROUTER}" \
      --seed "${SEED}" \
      --initial-checkpoint "${SUPERNET_CHECKPOINT}" \
      --pattern-budget-target "${TARGET}" \
      --pattern-budget-weight "${BUDGET_WEIGHT}" \
      "${ROUTER_FREEZE_ARGS[@]}" \
      --model-kwargs pattern_mode=router pattern_bank="${PATTERN_BANK}" router_init=zero_logits \
      "${COMMON_TRAIN_ARGS[@]}"
  done
fi

echo "Done. Use scripts/analyze_pattern_results.py to summarize output/train into results/analysis."
