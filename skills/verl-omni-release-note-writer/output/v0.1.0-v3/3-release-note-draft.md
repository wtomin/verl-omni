## Highlights

This release includes 133 merged PRs from 14 contributors.

VeRL-Omni `v0.1.0rc1` is the first dedicated release for multimodal generative RL training. It establishes the repository as a runnable stack for rollout, trainer, reward, and recipe development across diffusion, unified multimodal, and omni-modality models.

This is the first tagged release since multimodal generative RL training moved into a dedicated VeRL-Omni repository.

### Architecture

#### Rollout Engine

* Upgraded the rollout stack to vLLM-Omni v0.22.0 and aligned the release around the companion vLLM, vLLM-Omni, and verl dependency versions. **(#166, #167)**
* Improved rollout execution with step-wise batching co-execution, experimental rollout correction, and faster LoRA weight updates, reducing friction in multimodal generation workflows from trajectory generation through actor-to-rollout synchronization. **(#81, #93, #156)**
* Added configurable diffusion rollout attention backend selection with startup consistency validation against the actor attention backend, helping users catch mismatched rollout/training attention settings before large runs. **(#200)**

#### Training

* Moved the GPU actor path toward FA3 attention while preserving native and SDPA fallbacks when FA3 dependencies are unavailable. **(#141, #165)**
* Expanded trainer backend options with FSDP sequence parallelism and optional VeOmni actor/reference engines for diffusion training. **(#59, #104)**
* Added trainer-side observability and reproducibility tools, including diffusion MFU metrics and deterministic per-step and per-rollout seeding. **(#60, #128, #136)**

#### Reward

* Added multi-reward weighted aggregation so a run can combine multiple scoring signals through configurable reward functions and managers. **(#109)**
* Added external HTTP scorer support and async reward documentation, making it easier to serve expensive reward models separately and overlap reward scoring with rollout. **(#116, #155)**

### Model & Algorithm Supports

* Added the first Qwen-Image RL recipe set, covering FlowGRPO, Flow-DPPO, MixGRPO, GRPO-Guard, DiffusionNFT, and online DPO. These recipes make Qwen-Image the main text-to-image coverage anchor for the first release. **(#48, #58, #106, #126, #139, #164, #168)**
* Expanded diffusion generator coverage with SD3.5 DPO and FlowGRPO, plus Wan2.2 DanceGRPO for text-to-video training. **(#95, #98, #127, #142, #178)**
* Added BAGEL FlowGRPO support for unified understanding-and-generation training. **(#132, #137)**
* Added Qwen3-Omni Thinker GSPO + LoRA training with vLLM-Omni async rollout, giving the first release an omni-modality recipe alongside the diffusion and unified multimodal recipes. **(#113)**

### Hardware

* Added Ascend NPU support for Qwen-Image FlowGRPO, including NPU-oriented launch scripts and quickstart guidance for Atlas 800T A2-style setups. **(#68, #85, #181)**
* Extended NPU recipe coverage to Qwen-Image DiffusionNFT, Qwen-Image online DPO, and SD3.5 DPO. **(#127, #164, #174)**
* Documented NPU-specific runtime requirements and attention backend choices in the relevant installation and recipe guides. **(#68, #85, #181)**

### Documentation / Tooling

* Refreshed user-facing docs for installation, FlowGRPO, Ascend NPU, HTTP scorer services, async reward, multi-node training, diffusion MFU metrics, and algorithm-specific recipes. **(#128, #155, #167, #177, #181)**
* Simplified environment setup with project extras for GPU, rollout, training, OCR reward, and development workflows. When upgrading, refresh dependencies using the current install guide so the vLLM-Omni v0.22 stack and optional extras are installed together. **(#167)**
* Added CUDA Docker setup for users who prefer containerized environments, plus multi-node and larger-card-count examples for scaling Qwen-Image FlowGRPO beyond a single node. **(#177, #194, #195)**
* Expanded validation coverage with GPU smoke tests and e2e scripts for core diffusion and reward paths, including FlowGRPO, DPO, DiffusionNFT, Qwen3-Omni GSPO, and vLLM reward coverage. **(#45, #80, #127, #150)**
* Some examples include additional compatibility guidance, especially around the Qwen3-Omni GSPO stack and CUDA/flash-attn validation. Check the relevant example README before launching large runs. **(#113, #166)**

## Breaking Changes

* Adapted rollout integrations to the upstream `verl` `LLMServerClient` refactor. Custom rollout server/client code should migrate to the current rollout client and configuration paths before upgrading. **(#52)**
