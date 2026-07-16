#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NUM_GPUS="${NUM_GPUS:-4}"
DATA_DIR="${DATA_DIR:-${HOME}/data/qwen_image_single}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/debug_dumps}"
CURRENT_DIR="${DEBUG_DUMP_DIR:-${OUTPUT_ROOT}/current}"
BASELINE_DIR="${BASELINE_DIR:-${OUTPUT_ROOT}/baseline}"
CONSOLE_LOG="${CONSOLE_LOG:-${CURRENT_DIR}/console.log}"
MEMORY_LOG="${MEMORY_LOG:-${CURRENT_DIR}/memory.log}"
DEBUG_DUMP_STEPS="${DEBUG_DUMP_STEPS:-1,2}"
DEBUG_DUMP_MODE="${DEBUG_DUMP_MODE:-full}"
PERF_THRESHOLD="${PERF_THRESHOLD:-0.05}"

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DEBUG_DUMP_DIR="${CURRENT_DIR}"
export DEBUG_DUMP_STEPS
export DEBUG_DUMP_MODE

mkdir -p "${CURRENT_DIR}"
python "${SCRIPT_DIR}/create_single_sample_data.py" --output-dir "${DATA_DIR}"

memory_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >"${MEMORY_LOG}" &
  memory_pid="$!"
fi

cleanup() {
  if [[ -n "${memory_pid}" ]] && kill -0 "${memory_pid}" >/dev/null 2>&1; then
    kill "${memory_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

pushd "${REPO_ROOT}" >/dev/null

set +e
python "${SCRIPT_DIR}/run.py" \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=256 \
  data.seed=42 \
  data.shuffle=False \
  data.dataloader_num_workers=0 \
  data.train_max_samples=1 \
  data.val_max_samples=1 \
  actor_rollout_ref.model.algorithm=flow_grpo \
  actor_rollout_ref.model.path=Qwen/Qwen-Image \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.optim.weight_decay=0.0001 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=False \
  actor_rollout_ref.actor.data_loader_seed=42 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.seed=42 \
  actor_rollout_ref.actor.fsdp_config.full_determinism=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.04 \
  actor_rollout_ref.rollout.name=vllm_omni \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.pipeline.height=256 \
  actor_rollout_ref.rollout.pipeline.width=256 \
  actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
  actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
  actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
  actor_rollout_ref.rollout.algo.noise_level=1.0 \
  actor_rollout_ref.rollout.algo.sde_type=sde \
  actor_rollout_ref.rollout.algo.sde_window_size=null \
  actor_rollout_ref.rollout.algo.sde_window_range=null \
  actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
  actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility \
  reward.custom_reward_function.name=compute_score \
  reward.reward_model.enable=False \
  reward.num_workers=1 \
  trainer.logger='["console"]' \
  trainer.project_name=verl-nightly \
  trainer.experiment_name=qwen_image_flowgrpo_single_sample \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.total_training_steps=20 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir=null \
  trainer.validation_data_dir=null \
  2>&1 | tee "${CONSOLE_LOG}"
train_status="${PIPESTATUS[0]}"
set -e

if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi

python "${SCRIPT_DIR}/collect_metrics.py" \
  --console-log "${CONSOLE_LOG}" \
  --memory-log "${MEMORY_LOG}" \
  --out-json "${CURRENT_DIR}/metrics.json" \
  --steps "${DEBUG_DUMP_STEPS}" \
  --n-gpus "${NUM_GPUS}"

if [[ -d "${BASELINE_DIR}" ]]; then
  python "${SCRIPT_DIR}/compare_dumps.py" \
    --baseline-dir "${BASELINE_DIR}" \
    --current-dir "${CURRENT_DIR}" \
    --out-json "${CURRENT_DIR}/compare_report.json" \
    --steps "${DEBUG_DUMP_STEPS}"

  if [[ -f "${BASELINE_DIR}/metrics.json" ]]; then
    python "${SCRIPT_DIR}/collect_metrics.py" \
      --console-log "${CONSOLE_LOG}" \
      --memory-log "${MEMORY_LOG}" \
      --out-json "${CURRENT_DIR}/metrics.json" \
      --baseline-json "${BASELINE_DIR}/metrics.json" \
      --compare-report-json "${CURRENT_DIR}/perf_compare_report.json" \
      --steps "${DEBUG_DUMP_STEPS}" \
      --n-gpus "${NUM_GPUS}" \
      --threshold "${PERF_THRESHOLD}"
  fi
elif [[ "${BOOTSTRAP_BASELINE:-0}" == "1" ]]; then
  python "${SCRIPT_DIR}/compare_dumps.py" \
    --baseline-dir "${BASELINE_DIR}" \
    --current-dir "${CURRENT_DIR}" \
    --out-json "${CURRENT_DIR}/compare_report.json" \
    --steps "${DEBUG_DUMP_STEPS}" \
    --bootstrap-if-missing
  cp "${CURRENT_DIR}/metrics.json" "${BASELINE_DIR}/metrics.json"
else
  echo "No baseline at ${BASELINE_DIR}; skipping baseline comparisons."
fi

popd >/dev/null
