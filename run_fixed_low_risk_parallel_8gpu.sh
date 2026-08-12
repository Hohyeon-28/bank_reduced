#!/usr/bin/env bash
set -euo pipefail

# Launch four fixed low-risk pattern experiments in parallel.
#
# Default layout uses physical GPUs 2-9 as four 2-GPU DDP jobs:
#   low_risk_02 -> GPUs 2,3
#   low_risk_07 -> GPUs 4,5
#   low_risk_06 -> GPUs 6,7
#   low_risk_16 -> GPUs 8,9
#
# Override GPU_PAIRS or PORT_BASE if needed:
#   GPU_PAIRS="0,1;2,3;4,5;6,7" PORT_BASE=29600 bash run_fixed_low_risk_parallel_8gpu.sh

DEFAULT_DATA_DIR="${HOME}/shared/hdd_ext/nvme1/public/vision/classification/imageNet"
DATA_DIR="${1:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
PATTERN_LIST=(${PATTERN_LIST:-low_risk_02 low_risk_07 low_risk_06 low_risk_16})
GPU_PAIRS="${GPU_PAIRS:-2,3;4,5;6,7;8,9}"
PORT_BASE="${PORT_BASE:-29600}"
NPROC_PER_JOB="${NPROC_PER_JOB:-2}"

IFS=';' read -ra GPU_PAIR_ARRAY <<< "${GPU_PAIRS}"

if [[ "${#PATTERN_LIST[@]}" -ne "${#GPU_PAIR_ARRAY[@]}" ]]; then
  echo "PATTERN_LIST count (${#PATTERN_LIST[@]}) must match GPU_PAIRS count (${#GPU_PAIR_ARRAY[@]})." >&2
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "DATA_DIR=${DATA_DIR}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
echo "GPU_PAIRS=${GPU_PAIRS}"
echo "PORT_BASE=${PORT_BASE}"
echo "NPROC_PER_JOB=${NPROC_PER_JOB}"

pids=()
for i in "${!PATTERN_LIST[@]}"; do
  PATTERN="${PATTERN_LIST[$i]}"
  CUDA_IDS="${GPU_PAIR_ARRAY[$i]}"
  MASTER_PORT="$((PORT_BASE + i))"

  echo "[Launch] ${PATTERN} on GPUs ${CUDA_IDS}, MASTER_PORT=${MASTER_PORT}"
  PATTERNS="${PATTERN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_IDS}" \
  MASTER_PORT="${MASTER_PORT}" \
  NPROC="${NPROC_PER_JOB}" \
    bash run_fixed_low_risk_pattern_experiments.sh "${DATA_DIR}" "${CUDA_IDS}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
