#!/usr/bin/env bash
set -euo pipefail

# Average-budget adaptive MLP routing experiment.
#
# Motivation:
#   Train candidate patterns fairly with quota-balanced random sampling, then train a router
#   that keeps the average active MLP count near the target budget.
#
# Usage:
#   bash run_average_budget_experiments.sh [DATA_DIR] [GPU_IDS]
#
# Common overrides:
#   DATA_DIR=~/shared/hdd_ext/nvme1/public/vision/classification/imageNet bash run_average_budget_experiments.sh
#   CUDA_VISIBLE_DEVICES=2,8 bash run_average_budget_experiments.sh
#   RUN_BASELINE=0 bash run_average_budget_experiments.sh
#   EPOCHS_SUPERNET=300 EPOCHS_ROUTER=100 EPOCHS_JOINT=100 TARGET_BUDGETS="6" bash run_average_budget_experiments.sh
#   STAGE=baseline bash run_average_budget_experiments.sh
#   STAGE=router SUPERNET_CHECKPOINT=/path/to/model_best.pth.tar bash run_average_budget_experiments.sh
#   STAGE=router RUN_JOINT=0 SUPERNET_CHECKPOINT=/path/to/model_best.pth.tar bash run_average_budget_experiments.sh
#   STAGE=joint ROUTER_CHECKPOINT=/path/to/model_best.pth.tar bash run_average_budget_experiments.sh

DEFAULT_DATA_DIR="${HOME}/shared/hdd_ext/nvme1/public/vision/classification/imageNet"
DATA_DIR="${1:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
CUDA_IDS="${2:-${CUDA_VISIBLE_DEVICES:-2,8}}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WORKERS="${WORKERS:-8}"
EPOCHS_BASELINE="${EPOCHS_BASELINE:-300}"
EPOCHS_SUPERNET="${EPOCHS_SUPERNET:-300}"
EPOCHS_ROUTER="${EPOCHS_ROUTER:-100}"
EPOCHS_JOINT="${EPOCHS_JOINT:-100}"
TARGET_BUDGETS="${TARGET_BUDGETS:-6}"
BUDGET_WEIGHT="${BUDGET_WEIGHT:-0.01}"
PATTERN_BANK="${PATTERN_BANK:-configs/pattern_banks/random_budget6_bank12_seed42.yml}"
SUPERNET_EVAL_PATTERN="${SUPERNET_EVAL_PATTERN:-rand6_01}"
RESULTS_DIR="${RESULTS_DIR:-results/average_budget}"
STAGE="${STAGE:-all}"
RUN_BASELINE="${RUN_BASELINE:-1}"
FREEZE_ROUTER_BACKBONE="${FREEZE_ROUTER_BACKBONE:-1}"
RUN_JOINT="${RUN_JOINT:-1}"
JOINT_LR="${JOINT_LR:-1e-4}"
JOINT_WARMUP_EPOCHS="${JOINT_WARMUP_EPOCHS:-5}"

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_IDS}"
NPROC="${NPROC:-${#GPU_ARRAY[@]}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_IDS}"

mkdir -p "${RESULTS_DIR}"

echo "DATA_DIR=${DATA_DIR}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC=${NPROC}"
echo "STAGE=${STAGE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "PATTERN_BANK=${PATTERN_BANK}"
echo "SUPERNET_EVAL_PATTERN=${SUPERNET_EVAL_PATTERN}"

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

run_baseline() {
  echo "[Stage Z] ViT-S baseline training"
  torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
    --model vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS_BASELINE}" \
    --seed "${SEED}" \
    "${COMMON_TRAIN_ARGS[@]}"
}

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

if [[ "${STAGE}" == "baseline" ]]; then
  run_baseline
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "supernet" ]]; then
  echo "[Stage A] Uniform sampled supernet training"
  torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
    --model pattern_mlp_vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS_SUPERNET}" \
    --seed "${SEED}" \
    --model-kwargs pattern_mode=sampled_quota pattern_bank="${PATTERN_BANK}" fixed_pattern="${SUPERNET_EVAL_PATTERN}" \
    "${COMMON_TRAIN_ARGS[@]}"
fi

run_joint_finetune() {
  local target="$1"
  local checkpoint="$2"

  if [[ -z "${checkpoint}" ]]; then
    echo "ROUTER_CHECKPOINT is required for joint fine-tuning." >&2
    exit 2
  fi

  echo "[Stage D] Joint fine-tuning with average budget target ${target}"
  echo "Using router checkpoint: ${checkpoint}"
  printf '%s\n' "${checkpoint}" > "${RESULTS_DIR}/router_checkpoint_budget${target}.txt"

  torchrun --nproc_per_node="${NPROC}" train_sh.py "${DATA_DIR}" \
    --model pattern_mlp_vit_small_patch16_224_depth12 \
    --epochs "${EPOCHS_JOINT}" \
    --seed "${SEED}" \
    --initial-checkpoint "${checkpoint}" \
    --pattern-budget-target "${target}" \
    --pattern-budget-weight "${BUDGET_WEIGHT}" \
    --model-kwargs pattern_mode=router pattern_bank="${PATTERN_BANK}" router_init=zero_logits \
    "${COMMON_TRAIN_ARGS[@]}" \
    --lr "${JOINT_LR}" \
    --warmup-epochs "${JOINT_WARMUP_EPOCHS}"
}

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

    ROUTER_CHECKPOINT="$(find_latest_checkpoint || true)"
    if [[ "${RUN_JOINT}" == "1" ]]; then
      run_joint_finetune "${TARGET}" "${ROUTER_CHECKPOINT}"
    fi
  done
fi

if [[ "${STAGE}" == "joint" ]]; then
  if [[ -z "${ROUTER_CHECKPOINT:-}" ]]; then
    ROUTER_CHECKPOINT="$(find_latest_checkpoint || true)"
  fi
  for TARGET in ${TARGET_BUDGETS}; do
    run_joint_finetune "${TARGET}" "${ROUTER_CHECKPOINT}"
  done
fi

if [[ "${STAGE}" == "all" && "${RUN_BASELINE}" == "1" ]]; then
  run_baseline
fi

echo "Done. Use scripts/analyze_pattern_results.py to summarize output/train into results/analysis."
