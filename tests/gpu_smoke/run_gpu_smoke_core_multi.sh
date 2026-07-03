#!/usr/bin/env bash
# Multi-GPU core smoke tests that require multi-worker GPU topology.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "core-multi" 2 "$@"

run_test 0 "diffusion rollout seed multi-worker" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/agent_loop/test_diffusion_rollout_seed_gpu.py

run_test 1 "visual reward manager" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/reward_loop/test_visual_reward_manager.py

gpu_smoke_summary
