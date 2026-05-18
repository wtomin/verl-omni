# DPO Data Preparation

This example supports two DPO data paths:

- **online-DPO**: train from prompt-only data. The trainer samples images with
  the rollout model, scores them with the reward model, then selects win/lose
  pairs online.
- **offline-DPO**: pre-generate win/lose image pairs with a frozen reference
  model and reward model. The trainer consumes `{prompt, img_win, img_lose}`
  triples directly, skips rollout/reward/advantage computation, and reuses the
  same diffusion DPO loss.

## Online-DPO Prompt Data

Create UTF-8 text files with one prompt per line. File names are arbitrary:

```text
dataset/my_prompts/train_prompts.txt
dataset/my_prompts/eval_prompts.txt
```

Convert them to parquet:

```bash
python3 examples/dpo_trainer/data_process/prepare_online_dpo.py \
  --input_file dataset/my_prompts/train_prompts.txt \
  --output_file data/my_prompts/train.parquet \
  --data_source prompt_image_reward \
  --system_prompt "You are a helpful image generation assistant."

python3 examples/dpo_trainer/data_process/prepare_online_dpo.py \
  --input_file dataset/my_prompts/eval_prompts.txt \
  --output_file data/my_prompts/test.parquet \
  --split test \
  --data_source prompt_image_reward \
  --system_prompt "You are a helpful image generation assistant."
```

This writes:

- `data/my_prompts/train.parquet`
- `data/my_prompts/test.parquet`

Each row keeps the original prompt in `extra_info.raw_prompt` so custom reward
functions can score generated images against the prompt text.

Launch online-DPO with:

```bash
bash examples/dpo_trainer/run_sd35_medium_dpo_lora.sh \
  data.train_files=data/my_prompts/train.parquet \
  data.val_files=data/my_prompts/test.parquet
```

## Offline-DPO Pair Data

Offline-DPO first uses the reference model to generate multiple candidate
images per prompt, scores them with a reward function/model, and writes one
pre-ranked pair per prompt. The resulting parquet rows contain:

- `prompt`: chat-style prompt messages.
- `negative_prompt`: optional negative prompt messages.
- `img_win`: path to the highest-scoring generated image.
- `img_lose`: path to the lowest-scoring generated image.
- `win_score` and `lose_score`: reward scores used to order the pair.
- `extra_info.raw_prompt`: plain prompt text for traceability.

Generate offline pairs from any prompt file and choose the parquet output path
explicitly:

```bash
python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --input_file dataset/my_prompts/train_prompts.txt \
  --output_file data/offline_dpo/train.parquet \
  --image_dir data/offline_dpo/images/train \
  --model_path stabilityai/stable-diffusion-3.5-medium \
  --num_images_per_prompt 4 \
  --height 256 \
  --width 256 \
  --num_inference_steps 25 \
  --guidance_scale 4.0 \
  --reward_function_path verl_omni/utils/reward_score/unified_reward.py \
  --reward_function_name compute_score_unified_reward \
  --launch_reward_server \
  --reward_server_host 127.0.0.1 \
  --reward_server_port 8000 \
  --reward_model_name CodeGoat24/UnifiedReward-2.0-qwen3vl-8b

python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --input_file dataset/my_prompts/eval_prompts.txt \
  --output_file data/offline_dpo/test.parquet \
  --image_dir data/offline_dpo/images/test \
  --split test \
  --model_path stabilityai/stable-diffusion-3.5-medium \
  --num_images_per_prompt 4 \
  --height 256 \
  --width 256 \
  --num_inference_steps 25 \
  --guidance_scale 4.0 \
  --reward_function_path verl_omni/utils/reward_score/unified_reward.py \
  --reward_function_name compute_score_unified_reward \
  --launch_reward_server \
  --reward_server_host 127.0.0.1 \
  --reward_server_port 8000 \
  --reward_model_name CodeGoat24/UnifiedReward-2.0-qwen3vl-8b
```

`--launch_reward_server` starts a `vllm serve` subprocess with the reward model
and waits for `/v1/models` before scoring. If you already have an
OpenAI-compatible reward server running, omit `--launch_reward_server` and pass
`--reward_router_address host:port` instead. For custom vLLM flags, override
`--reward_server_command`; the template can use `{model}`, `{host}` and
`{port}`.

This writes:

- `data/offline_dpo/train.parquet`
- `data/offline_dpo/test.parquet`
- generated images under the requested `--image_dir`

Train on the offline triples with:

```bash
bash examples/dpo_trainer/run_sd35_medium_offline_dpo_lora.sh \
  data.train_files=data/offline_dpo/train.parquet \
  data.val_files=data/offline_dpo/test.parquet
```

During offline training, `offline_dpo_trainer.yaml` sets
`algorithm.dpo_mode=offline` and `data.offline_dpo=true`. The dataset expands
each row into adjacent `[win, lose]` samples with a shared `uid`, and the trainer
materializes `image_latents` plus SD3 text-encoder prompt embeddings before
calling the existing DPO loss.

## Reward Template

`examples/dpo_trainer/reward_score/prompt_image_reward.py` shows the expected
custom reward interface. It ignores `ground_truth` and sends the generated image
plus `extra_info.raw_prompt` to an OpenAI-compatible vision reward model.

Example config overrides:

```bash
data=legacy_data \
data.train_files=data/my_prompts/train.parquet \
data.val_files=data/my_prompts/test.parquet \
reward.custom_reward_function.path=examples/dpo_trainer/reward_score/prompt_image_reward.py \
reward.custom_reward_function.name=compute_score
```

`dpo_trainer.yaml` uses `legacy_data` by default for online-DPO and adds the
DPO-specific fields there, including `negative_prompt_key`, `gen_batch_size`,
and `k_samples`. `offline_dpo_trainer.yaml` inherits the same DPO setup but
switches to offline pair loading and disables training-time reward workers by
default.
