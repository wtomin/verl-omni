## Highlights Draft

This release includes 133 merged PRs from 14 contributors.

VeRL-Omni `v0.1.0` is the first dedicated release for multimodal generative RL training. It establishes the repository as a runnable stack for rollout, trainer, reward, and recipe development across diffusion, unified multimodal, and omni-modality models.

This is the first tagged release since multimodal generative RL training moved into a dedicated VeRL-Omni repository.

### Architecture

#### Rollout Engine

* Aligns rollout on vLLM-Omni v0.22.0 and improves rollout execution with step-wise batching co-execution, rollout correction, and faster LoRA weight updates. **(#81, #93, #156, #166)**

#### Training

* Strengthens the training architecture with FA3 actor attention and fallbacks, FSDP sequence parallelism, optional VeOmni engines, diffusion MFU metrics, and deterministic seeding. **(#59, #60, #104, #128, #141, #165)**

#### Reward

* Adds multi-reward weighted aggregation, HTTP scorer integration, and async reward guidance for flexible reward serving and reward-over-rollout overlap. **(#109, #116, #155)**

### Model & Algorithm Supports

* Adds the first broad recipe set: Qwen-Image FlowGRPO, Flow-DPPO, MixGRPO, GRPO-Guard, DiffusionNFT, and online DPO; SD3.5 DPO and FlowGRPO; Wan2.2 DanceGRPO; BAGEL FlowGRPO; and Qwen3-Omni Thinker GSPO + LoRA. **(#48, #58, #95, #98, #113, #126, #127, #137, #139, #164, #168, #178)**

### Hardware

* Adds Ascend NPU coverage for Qwen-Image FlowGRPO, DiffusionNFT, online DPO, and SD3.5 DPO, with restored install guidance and NPU-oriented recipe scripts. **(#68, #85, #127, #164, #174, #181)**

### Documentation / Tooling

* Improves adoption through installation extras, CUDA Docker setup, multi-node and 64-card recipe docs, GPU smoke tests, and e2e coverage for core diffusion/reward paths. **(#45, #80, #167, #177, #194, #195)**
