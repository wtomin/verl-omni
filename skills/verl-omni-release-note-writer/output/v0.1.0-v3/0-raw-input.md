# VeRL-Omni v0.1.0-v3 Raw Input

## Release Boundary

- Current tag: `v0.1.0` draft
- Previous tag: `initial`
- Release type: final release draft
- This is treated as the first tagged release for the dedicated `verl-project/verl-omni` repository.

## Source Material

- Updated release-note writer skill and references
- Existing `output/v0.1.0` and `output/v0.1.0-v2` artifacts
- `README.md` News, scope, model x algorithm support table, and Ascend NPU section
- Key example and documentation pages for GSPO, install, NPU quickstart, HTTP scorer, and async reward

## v3 Editorial Changes

- Use the requested roadmap-style directory:
  - `Architecture`
  - `Architecture / Rollout Engine`
  - `Architecture / Training`
  - `Architecture / Reward`
  - `Model & Algorithm Supports`
- Do not create a separate non-breaking caveats section.
- Fold install and compatibility reminders into `Documentation / Tooling`.
- Keep WIP and Planned recipe caveats inside `Model & Algorithm Supports`.
- Avoid dense interleaving of prose with commands, paths, package pins, and config keys. Exact install commands and version details are grouped instead of scattered through the body.

## Release Story

VeRL-Omni `v0.1.0` is the first release for a dedicated multimodal generative RL training repository. It packages the main architecture work around rollout and trainer backends together with verified model x algorithm recipes for image, video, unified multimodal, and omni-modality training.

## Representative User-Facing PRs

- `#113` Qwen3-Omni Thinker GSPO + LoRA with vLLM-Omni async rollout
- `#168` Flow-DPPO support for Qwen-Image
- `#106`, `#164` DiffusionNFT support and Qwen-Image recipe/NPU coverage
- `#139`, `#174` Qwen-Image online DPO and NPU recipe
- `#95`, `#127`, `#178` SD3/SD3.5 DPO and FlowGRPO coverage
- `#98`, `#142` Wan2.2 DanceGRPO text-to-video training and dataset docs
- `#58`, `#126` MixGRPO and Qwen-Image recipes
- `#48`, `#126` GRPO-Guard and Qwen-Image recipes
- `#132`, `#137` BAGEL FlowGRPO integration
- `#166` vLLM-Omni v0.22.0 alignment
- `#165`, `#141` FA3 actor attention backend support and fallback behavior
- `#59`, `#104` FSDP sequence parallelism and VeOmni actor/reference engine support
- `#60`, `#93`, `#136` deterministic seeding and rollout correction
- `#109`, `#116`, `#155` multi-reward, HTTP scorer, and async reward documentation
- `#68`, `#85`, `#127`, `#164`, `#174`, `#181` Ascend NPU recipes and install documentation
- `#128` diffusion MFU metrics
- `#167` `pyproject.toml` and `uv` extras installation
- `#177`, `#194` multi-node and larger-card-count recipe docs
- `#45`, `#80`, `#150` smoke/e2e/CI stability coverage
- `#195` CUDA Dockerfile and install guide
- `#52` breaking rollout client adaptation to upstream `verl` `LLMServerClient` refactor
