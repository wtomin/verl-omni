#!/usr/bin/env bash
# Single-GPU core smoke tests: rollout, agent loop, and diffusion engine coverage.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "core-single" 1 "$@"

run_test 0 "vllm-omni rollout" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py

run_test 1 "diffusion agent loop" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/agent_loop/test_diffusion_agent_loop.py

run_test 2 "diffusers FSDP engine" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/test_diffusers_fsdp_engine.py

# Skips itself if the optional `veomni` backend is not installed (importorskip).
run_test 3 "diffusers VeOmni engine" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/test_diffusers_veomni_engine.py

gpu_smoke_summary
