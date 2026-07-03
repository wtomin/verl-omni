#!/usr/bin/env bash
# Reward-model colocated trainer e2e smoke tests.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "reward-e2e" 4 "$@"

run_test 0 "FlowGRPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image.sh

run_test 1 "Qwen-Image online DPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_online_dpo_qwen_image.sh

gpu_smoke_summary
