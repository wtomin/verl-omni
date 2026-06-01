# DPO Training

This example supports offline DPO training for diffusion models. The data
preparation step first generates several candidate images per prompt with a
frozen reference pipeline, scores the candidates, and writes one pre-ranked
win/lose pair per prompt. Training consumes those pairs directly and does not
run online rollout, training-time reward scoring, or online pair selection.

## Pair Data

The resulting parquet rows contain:

- `prompt`: chat-style prompt messages.
- `negative_prompt`: optional negative prompt messages.
- `img_win`: path to the highest-scoring generated image.
- `img_lose`: path to the lowest-scoring generated image.
- `img_win_latents` and `img_lose_latents`: precomputed diffusion latents
  formatted for the selected pipeline. Qwen-Image stores packed latents.
- `prompt_embeds` and `prompt_embeds_mask`: precomputed text-encoder outputs.
- `pooled_prompt_embeds`: optional SD3-only pooled text-encoder output.
- `win_score` and `lose_score`: reward scores used to order the pair.
- `extra_info.raw_prompt`: plain prompt text for traceability.

## Qwen-Image Training

Generate Qwen-Image offline pairs from prompt files and choose the parquet
output paths explicitly:

```bash
python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --pipeline qwen_image \
  --input_file dataset/my_prompts/train_prompts.txt \
  --output_file data/offline_dpo_qwen_image/train.parquet \
  --image_dir data/offline_dpo_qwen_image/images/train \
  --num_images_per_prompt 8 \
  --height 512 \
  --width 512 \
  --num_inference_steps 35 \
  --true_cfg_scale 4.0 \
  --reward_function_path verl_omni/utils/reward_score/unified_reward.py \
  --reward_function_name compute_score_unified_reward \
  --launch_reward_server \
  --reward_server_host 127.0.0.1 \
  --reward_server_port 8000 \
  --reward_model_name CodeGoat24/UnifiedReward-2.0-qwen3vl-8b

python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --pipeline qwen_image \
  --input_file dataset/my_prompts/eval_prompts.txt \
  --output_file data/offline_dpo_qwen_image/test.parquet \
  --image_dir data/offline_dpo_qwen_image/images/test \
  --split test \
  --num_images_per_prompt 8 \
  --height 512 \
  --width 512 \
  --num_inference_steps 35 \
  --true_cfg_scale 4.0 \
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

- `data/offline_dpo_qwen_image/train.parquet`
- `data/offline_dpo_qwen_image/test.parquet`
- generated images under the requested `--image_dir`

Train on the offline triples with:

```bash
bash examples/dpo_trainer/run_qwen_image_offline_dpo_lora.sh \
  data.train_files=data/offline_dpo_qwen_image/train.parquet \
  data.val_files=data/offline_dpo_qwen_image/test.parquet
```

During training, `run_sd35_medium_offline_dpo_lora.sh` sets `algorithm.sample_source=offline`
and loads `OfflineDPODataset` via `data.custom_cls`. The dataset expands each row into adjacent
`[win, lose]` samples with a shared `uid`. Collate stacks the precomputed
`image_latents` plus SD3 prompt embeddings from parquet before calling the DPO
loss, so training does not load the SD3 VAE or text encoders during
actor updates. Offline DPO also disables rollout and reward workers, so
validation generation is disabled by default.

### Sample Filtering

You can reject weak win/lose pairs at dataloader sampling time. When a sampled
row fails the filter, the dataset resamples until it finds a qualifying row.
Configure a callable through `data.sample_filter_fn`:

```bash
bash examples/dpo_trainer/run_qwen_image_offline_dpo_lora.sh \
  data.sample_filter_fn.path=pkg://verl_omni.utils.dataset.offline_dpo_dataset \
  data.sample_filter_fn.name=offline_dpo_score_gap_filter \
  data.sample_filter_fn.min_score_gap=0.07
```

The built-in `offline_dpo_score_gap_filter` keeps rows where
`win_score - lose_score > min_score_gap` (default `0.07`). You can also point
`path`/`name` to your own function that accepts a parquet row dict and returns
`True` when the row should be kept.

## SD3.5 Training

Stable Diffusion 3.5 is also supported. Use `--pipeline sd3` or leave
`--pipeline` as `auto` when `--model_path` points to an SD3 model:

```bash
python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --pipeline sd3 \
  --input_file dataset/my_prompts/train_prompts.txt \
  --output_file data/offline_dpo_sd3/train.parquet \
  --image_dir data/offline_dpo_sd3/images/train \
  --model_path stabilityai/stable-diffusion-3.5-medium \
  --num_images_per_prompt 8 \
  --height 256 \
  --width 256 \
  --num_inference_steps 25 \
  --guidance_scale 4.0
```

Train SD3.5 pairs with:

```bash
bash examples/dpo_trainer/run_sd35_medium_offline_dpo_lora.sh \
  data.train_files=data/offline_dpo_sd3/train.parquet \
  data.val_files=data/offline_dpo_sd3/test.parquet
```

## Offline Checkpoint Validation

After training, validate saved checkpoints on a prompt text file and plot
training step vs. reward score:

```bash
python3 examples/dpo_trainer/validate_offline_dpo_checkpoints.py \
  --pipeline qwen_image \
  --model_path Qwen/Qwen-Image \
  --prompt_file dataset/my_prompts/validation_prompts.txt \
  --checkpoint_dir checkpoints/offline_dpo/qwen_image_offline_dpo_lora \
  --output_dir outputs/offline_dpo_validation/qwen_image \
  --reward_router_address 127.0.0.1:8000 \
  --reward_model_name CodeGoat24/UnifiedReward-2.0-qwen3vl-8b \
  --height 512 \
  --width 512 \
  --num_inference_steps 35 \
  --true_cfg_scale 4.0
```

The script discovers `global_step_*/actor` checkpoints under
`--checkpoint_dir`. You can also pass explicit checkpoint files or directories
through `--checkpoint_paths`. For each checkpoint, it generates images for the
validation prompts, calls the reward function, writes `validation_summary.jsonl`,
`validation_details.jsonl`, CSV copies of both files, and saves
`reward_curve.png`.

When an actor directory contains multi-GPU FSDP shards such as
`model_world_size_4_rank_*.pt`, the script merges them into a full checkpoint
before loading. By default (`--fsdp_merge_backend auto`) it uses
[`verl.model_merger`](https://verl.readthedocs.io/en/latest/advance/checkpoint.html)
merge logic in-process and falls back to a local merge for diffusion LoRA
checkpoints. To export a HuggingFace-format checkpoint manually, use:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/offline_dpo/qwen_image_offline_dpo_lora/global_step_30/actor \
  --target_dir /path/to/merged_hf_model \
  --trust-remote-code
```

Or pass `--fsdp_merge_backend verl_cli` to run the same command inside the
validation script (mainly for standard HuggingFace actor checkpoints).

## Reward Template

`prepare_offline_dpo.py` can call any reward function with the standard
VeRL-Omni custom reward signature. The example command above uses
`verl_omni/utils/reward_score/unified_reward.py` and can either launch a local
OpenAI-compatible vLLM reward server or connect to an existing one through
`--reward_router_address`.
