#!/usr/bin/env bash
# Qwen3-Omni VeOmni-native VLM DPO smoke test.
#
# This follows VeOmni's VLM training style:
#
#   bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_omni/qwen3_omni.yaml
#
# but uses a local smoke task that calls VeOmni's VLMTrainer components and
# applies a DPO loss to tiny Omni-Preference-style multimodal preference data.
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_omni_preference_dpo}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
RUN_DIR=${RUN_DIR:-${DATA_DIR}/veomni_vlm_dpo_smoke}
VEOMNI_ROOT=${VEOMNI_ROOT:-}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_SCRIPT="${REPO_ROOT}/tests/special_e2e/veomni_train_vlm_dpo_smoke.py"

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("veomni") is not None else 1)
PY
then
    echo "Skipping Qwen3-Omni VeOmni VLM DPO smoke: optional package \`veomni\` is not installed." >&2
    exit 0
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Resolve the tiny model ─────────────────────────────────────────────────────
if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="${HOME}/models/tiny-random/Qwen3-Omni-mllm"
    [ -d "${MODEL_PATH}" ] || python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
        --output-dir "${MODEL_PATH}"
fi

# ── Build dummy preference data ────────────────────────────────────────────────
python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_omni_preference_dpo_data.py" \
    --local_save_dir "${DATA_DIR}"

mkdir -p "${RUN_DIR}"
DATA_CONFIG="${RUN_DIR}/omni_preference_dpo_multisource.yaml"
TRAIN_CONFIG="${RUN_DIR}/qwen3_omni_veomni_vlm_dpo_smoke.yaml"

cat >"${DATA_CONFIG}" <<EOF
sources:
- ${DATA_DIR}/image/train.parquet
- ${DATA_DIR}/video/train.parquet
- ${DATA_DIR}/audio/train.parquet
names:
- Omni-Preference-Image
- Omni-Preference-Video
- Omni-Preference-Audio
schedule:
- schedule_type: const
  weights: [0.34, 0.33, 0.33]
EOF

cat >"${TRAIN_CONFIG}" <<EOF
model:
  model_path: ${MODEL_PATH}
  tokenizer_path: ${MODEL_PATH}
  ops_implementation:
    attn_implementation: sdpa
    moe_implementation: eager
    cross_entropy_loss_implementation: eager
    rms_norm_implementation: eager
    swiglu_mlp_implementation: eager
    rotary_pos_emb_implementation: eager
    load_balancing_loss_implementation: eager

data:
  train_path: ${DATA_CONFIG}
  data_type: dpo
  max_seq_len: 512
  train_size: 6
  dataloader:
    num_workers: 0
    use_background_prefetcher: false
  mm_configs:
    scale_factor: 28
    image_min_pixels: 3136
    image_max_pixels: 12845056
    video_min_pixels: 3136
    video_max_pixels: 602112
    max_ratio: 200
    min_frames: 2
    max_frames: 4
    frame_factor: 1
    sample_rate: 16000
    fps: 2.0
    use_audio_in_video: false

train:
  accelerator:
    ulysses_size: 1
    fsdp_config:
      fsdp_mode: ddp
      mixed_precision:
        enable: true
  gradient_checkpointing:
    enable: true
  optimizer:
    type: adamw
    lr: 1.0e-6
    lr_decay_style: cosine
    lr_warmup_ratio: 0.0
    max_grad_norm: 1.0
  num_train_epochs: 1
  micro_batch_size: 1
  global_batch_size: 2
  max_steps: ${TOTAL_TRAIN_STEPS}
  init_device: cuda
  checkpoint:
    output_dir: ${RUN_DIR}/checkpoints
    save_steps: 1000
    save_hf_weights: false
  wandb:
    project: verl-test
    name: qwen3-omni-veomni-vlm-dpo-smoke

dpo_config:
  beta: 0.1
  label_smoothing: 0.0
  loss_type: sigmoid
  reference_free: false
  refer_model_precision: bfloat16
EOF

if [ -n "${VEOMNI_ROOT}" ] && [ -f "${VEOMNI_ROOT}/train.sh" ]; then
    bash "${VEOMNI_ROOT}/train.sh" "${TASK_SCRIPT}" "${TRAIN_CONFIG}" "$@"
else
    torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${TASK_SCRIPT}" "${TRAIN_CONFIG}" "$@"
fi

echo "Qwen3-Omni VeOmni VLM DPO smoke test passed."

