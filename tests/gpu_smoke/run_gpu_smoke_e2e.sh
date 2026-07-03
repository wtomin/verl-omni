#!/usr/bin/env bash
# Trainer e2e smoke tests without a colocated reward model.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "e2e" 2 "$@"

run_test 0 "DiffusionNFT trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_diffusionnft_qwen_image.sh

# Fixed at 2 GPUs: the smoke stage config pins tensor_parallel_size=2 and FSDP
# needs >1 GPU to shard (NO_SHARD can't run the offload_to_cpu LoRA-sync summon).
run_test 1 "Qwen3-Omni Thinker GSPO LoRA e2e" \
    env CUDA_VISIBLE_DEVICES="0,1" NUM_GPUS=2 \
    bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_smoke.sh

gpu_smoke_summary
