# VeRL-Omni v0.2.0: Faster Diffusion RL and Stable Omni Training

> Estimated reading time: 8 minutes.
> Focus areas: faster diffusion RL and stable omni training.

VeRL-Omni `v0.2.0` establishes a stronger foundation for production-grade omni-modal reinforcement learning. This release improves the training stack across rollout performance, model integration, reward support, hardware coverage, and documentation, with two changes carrying the most impact:

- Faster diffusion RL, centered on higher-throughput Qwen-Image FlowGRPO rollout and V1 trainer support.
- Stable omni training, built around the omni V1 trainer, reusable model adapters, FSDP2, and vLLM-Omni rollout.

<div align="center">
  <img
    src="assets/verl_omni_v0_2_0_blog_overview.png"
    alt="VeRL-Omni v0.2.0 release overview"
    width="60%"
  />
</div>

## 1. Faster Diffusion RL

Diffusion RL is expensive, but not in the same way as autoregressive language-model RL. A single rollout carries many denoising steps, large latent tensors, prompt embeddings, optional classifier-free guidance, reward-model scoring, old-log-prob recomputation, and policy-weight synchronization. For Qwen-Image FlowGRPO, there is no single villain in the profile. Step time is shaped by rollout generation, old-log-prob computation, reward scoring, actor update, and LoRA weight sync together.

### Key Features

The faster diffusion RL work has two main features.

- Request-level batching leads the rollout side. For supported diffusion adapters, it becomes the default batching path. Instead of sending diffusion generations through a serial loop, the engine now acts more like a traffic controller, with explicit concurrency knobs for scheduling rollout work.

- The trainer path matters just as much. Diffusion now has a V1 trainer path, bringing diffusion RL closer to the modern trainer architecture used elsewhere in VeRL-Omni and laying the groundwork for decoupled rollout and training execution.

Faster rollout only matters if the generated trajectories and log-probs still describe the same policy. This release fixes several correctness-sensitive areas: request-batched diffusion log-probs, async rollout semantics, and rank-local LoRA weight-update routes. These details are the rails that keep high-throughput rollout trainable.

### Current Repository Support

The [rollout batching guide](https://verl-omni.readthedocs.io/en/latest/start/rollout_batching.html) backs up the rollout story. It explains both diffusion batching modes, how to enable them, and when to choose each mode. It also reports request-level batching gains:

- Qwen-Image LoRA, 32 prompts × 16 responses, 512 px, `max_num_seqs=32`: generation time drops from `226.4s` to `107.9s`, a `52%` reduction.
- SD3.5 LoRA, 8 prompts × 8 responses, 384 px, `max_num_seqs=256`: generation time drops from `25.4s` to `22.3s`, a `12%` reduction.

On the trainer side, diffusion V1 support currently focuses on SD3.5 FlowGRPO. The repository includes both a synchronous V1 recipe and a V1 `separate_async` recipe, where dedicated rollout workers can overlap generation with training and improve throughput compared with the legacy v0 trainer path.

### Recipe and Benchmark

Production-style Qwen-Image FlowGRPO recipes enable request-level batching by default and cover several common setups:

- `run_qwen_image_ocr_lora.sh`: Qwen-Image LoRA FlowGRPO with OCR reward.
- `run_qwen_image_ocr_lora_async_reward.sh`: async reward on a dedicated resource pool.
- `run_qwen_image_ocr_lora_rollout_corr.sh`: rollout-correction bypass mode.

Current reference numbers include:

- Qwen-Image FlowGRPO LoRA on 4 × H800: `420s` per step.
- Async reward variant on 5 GPUs:  `360s` per step.
- Rollout-correction bypass mode skips actor old-log-prob recomputation and saves about `20%` per-step time.

[add some graphs]

SD3.5 FlowGRPO also has V1 trainer coverage:

- `run_sd35_medium_ocr_lora_v1.sh`: SD3.5 LoRA FlowGRPO with the V1 trainer in sync mode.
- `run_sd35_medium_ocr_lora_v1_separate_async.sh`: SD3.5 LoRA FlowGRPO with V1 `separate_async` rollout.


[add some graphs and provide statistics]

## 2. Stable Omni Training

The other half of the release is stable omni training. Omni models are not just bigger language models; they are small systems with processors, modality-specific towers, trainable stages, and rollout-time behavior that has to stay aligned with the actor. `v0.2.0` moves the project from model-specific integrations toward a reusable omni training stack, so multimodal autoregressive training fits more naturally into VeRL-Omni's trainer, adapter, rollout, and recipe structure.

### Key Features

Here, the release pulls on two levers.

One lever is the `verl` V1 trainer architecture. Omni recipes get clearer worker orchestration, standard configuration overrides, and better alignment with vLLM-Omni rollout.

The other is the reusable omni model adapter layer. Instead of wiring each architecture as a one-off path, the trainer can rely on a shared interface for model setup, processor setup, trainable-stage selection, FSDP preparation, and rollout alignment.

### Current Repository Support

The current Qwen3-Omni adapter supports thinker-only training by redirecting training to the target component, stripping unused modules such as Talker and codec-related components, and working with FSDP/FSDP2 wrapping. This keeps inactive components out of the sharded training graph while preserving the multimodal processor and rollout interface.

Supported algorithms:

- **GSPO** for online RL-style training of the omni Thinker path. This is the most complete path today and is the main training flow for the released recipes.
- **Omni DPO** for offline multimodal preference training. The release adds the offline MLLM preference dataset pipeline, Omni DPO config, and `OmniDPOLoss`.

Supported multimodal training modes:

- **Text -> text** reasoning through the GSM8K GSPO recipe.
- **Image -> text** reasoning through the MMK12 GSPO recipe.
- **Text + image + audio -> text** reasoning through the AVQA-R1-6K NPU recipe.
- **Offline multimodal preference training** through the omni DPO data pipeline and trainer path.

The V1 omni launchers also document healthy signals that users can check during training:

- `training/rollout_actor_probs_pearson_corr` above `0.995`, showing actor and rollout agreement after weight sync.
- `rollout_corr/log_ppl_diff` near zero, showing rollout and actor log-prob consistency.
- Stable actor loss and gradient norm ranges.
- Validation accuracy or reward rising with training steps.

These signals matter because stable omni training needs evidence of correctness, not only a successful launch command.

### Recipe and Benchmark

The best single recipe to highlight is **MMK12**. It exercises the new stable Qwen3-Omni path with real multimodal input: image plus text prompt, text answer, GSPO optimization, FSDP actor training, and vLLM-Omni rollout.

**MMK12 anchor recipe.** `run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh` trains Qwen3-Omni on K12 visual math reasoning (`image -> text`) with GSPO, LoRA rank 32, and colocated actor-rollout workers on 4 × H800 80GB. The rollout shape is 128 prompts × 16 responses, or 2048 samples per rollout. After training, the run reaches `0.833` validation reward, `0.998` actor-rollout Pearson correlation, and about `59 GB` GPU memory usage. See some training results in the reference run: [`MMK12 (wandb)`](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36).

<div style="display: flex; gap: 16px; justify-content: center; align-items: flex-start;">
  <div style="width: 45%; text-align: center;">
    <img
      src="assets/mmk12_training_rewards.svg"
      alt="MMK12 training rewards mean scores"
      width="100%"
    />
    <p style="margin-top: 8px; text-align: center;"><em>MMK12 training rewards mean scores.</em></p>
  </div>
  <div style="width: 45%; text-align: center;">
    <img
      src="assets/mmk12_val_rewards.svg"
      alt="MMK12 validation rewards mean scores"
      width="100%"
    />
    <p style="margin-top: 8px; text-align: center;"><em>MMK12 validation rewards mean scores.</em></p>
  </div>
</div>

The MMK12 data pipeline converts raw MMK12 parquet shards into the verl RL parquet layout. Each row carries the image bytes inline and uses a prompt format that asks the model to produce a structured answer. The reward combines `math_verify` accuracy with a progressive format reward on the `<answer>...\boxed{}...</answer>` template.

To run the recipe:

```bash
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path /path/to/mmk12/ \
    --local_save_dir ~/data/mmk12

TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh
```

This anchors the `v0.2.0` stability story: Qwen3-Omni training is no longer just a model-specific launch path. It is a V1 trainer recipe with a reusable omni adapter, multimodal data handling, actor-rollout consistency metrics, and a documented image-to-text benchmark.

## Briefly: Other Updates

The release also expands the broader VeRL-Omni surface:

- LTX-2.3 text-to-audio-video FlowGRPO with CLAP and ImageBind rewards.
- Qwen-Image-Edit FlowGRPO and a general image-editing interface.
- BAGEL full-parameter training with PickScore reward
- DiNa-LRM reward support for SD3.5 FlowGRPO.
- Ascend NPU Dockerfiles and install guide.

## Looking Ahead

The road after `v0.2.0` is fairly clear.

For diffusion RL, the next step is to keep moving from "works end-to-end" to "measurably efficient and diagnosable". The repository now has the features for this: batching modes, profiling recipes, and rollout-correction paths. Future benchmark suites should separate rollout, log-prob computation, reward scoring, weight sync, and actor update costs.

For omni model training, the adapter design should become a reusable model-integration pattern. It points toward more omni models with stage-specific training, for example, minicpm-o.

Correctness signals should keep getting tighter. For both diffusion and omni training, high throughput is only useful when log-probs, rollout trajectories, actor weights, and reward signals remain aligned. Metrics such as actor-rollout Pearson correlation, log-prob consistency, precision dumps, and step-level timing should become standard parts of every serious recipe.

Reproducibility should also become a first-class release goal. Future recipes should make it easier to rerun the same setup with pinned model versions, fixed data preprocessing, explicit seeds, stable configuration overrides, and published reference metrics. For long-running RL jobs, reproducibility is not only about getting the same final score; it is about making reward curves, timing breakdowns, memory usage, and validation outputs comparable across machines and releases.

