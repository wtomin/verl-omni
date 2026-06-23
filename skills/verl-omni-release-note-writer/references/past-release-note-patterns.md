# Historical VeRL-Omni Release Note Patterns

This reference defines the editorial style for `verl-project/verl-omni` release notes. The project is young and may not yet have many GitHub Releases; use this document as the baseline until real releases accumulate.

## Cold-Start Sources

Until tagged releases exist, derive patterns from:

- README `News` section
- README model x algorithm support table
- `examples/*/README.md` verified recipes
- `docs/algo/*.md` and `docs/start/*.md`
- parent project `verl` release notes for section hierarchy, tone, and Breaking Changes handling

Do not copy vLLM-Omni inference-centric section names. VeRL-Omni release notes should follow the roadmap-style directory centered on `Architecture` and `Model & Algorithm Supports`.

## Default Style Rules

- Start with the release story, not raw PR enumeration
- Organize by user-facing roadmap domains, not code directories
- Merge related PRs into one sentence when they support the same claim
- Mention representative PR numbers after the claim, keeping citations bolded as `**(#1234)**`
- Pair algorithms with model families and example entry points
- Keep internal refactors only when they unlock visible training or rollout capability
- RC and final releases share the same section structure

## Section Bank

Use these roadmap-style body sections. Omit any section or subsection with no substantive user-facing content.

### Architecture

Purpose: how the rollout engine and trainer architecture changed.

#### Rollout Engine

Typical content:

- vLLM-Omni version upgrades and async rollout
- rollout adapters, servers, batching, and LoRA weight update paths
- deterministic rollout behavior
- rollout correction
- rollout throughput or latency improvements

Example phrasing:

- "Upgraded vLLM-Omni rollout backend to v0.22 for higher throughput."
- "Optimized LoRA rollout weight updates for faster actor-to-rollout synchronization."

#### Training

Typical content:

- FSDP2 and VeOmni engine support
- USP/TP/DP parallelism
- deterministic trainer behavior
- default attention backend changes such as FA3
- training throughput, MFU, memory, or stability improvements tied to backend work

Example phrasing:

- "Expanded FSDP2 and VeOmni backend coverage for diffusion actor training."
- "Added diffusion MFU metrics for comparing training configurations on the same setup."

#### Reward

Typical content:

- multi-reward managers
- weighted reward aggregation
- HTTP scorer integration and external reward services
- asynchronous reward computation and reward-loop overlap
- reward-loop correctness, latency, or determinism fixes

Example phrasing:

- "Added asynchronous reward computation to overlap rollout on a dedicated GPU pool."
- "Extended multi-reward serving with UnifiedReward and GenRM-OCR scorers."

### Model & Algorithm Supports

Purpose: what new model x algorithm workflows users can run.

Typical content:

- FlowGRPO, Flow-DPPO, MixGRPO, GRPO-Guard, DiffusionNFT, DPO, DanceGRPO, GSPO
- Qwen-Image, Wan2.2, BAGEL, Qwen3-Omni, SD3.5 pipelines
- X2I, X2V, X2A, unified understanding/generation, and omni model groupings
- links to `examples/` and algorithm docs

Example phrasing:

- "Added verified DiffusionNFT and Diffusion DPO recipes for Qwen-Image and SD3.5."
- "Introduced Qwen3-Omni GSPO trainer with example scripts under `examples/gspo_trainer/`."
- "Added Wan2.2 DanceGRPO video generation support."

Reward changes that only expose new scorer choices for a recipe may be mentioned here; reward-loop or serving architecture belongs under `Architecture / Reward`.

### Hardware

Purpose: platform-specific backend support beyond generic GPU training.

Typical content:

- Ascend NPU support and NPU quickstart
- NPU e2e scripts under `tests/special_e2e/`
- hardware-specific runtime caveats

Example phrasing:

- "Added Ascend NPU support with FlowGRPO e2e coverage and NPU quickstart documentation."

### Documentation / Tooling

Purpose: remaining release-facing work that helps users adopt or validate the release.

Typical content:

- docs additions or major refreshes
- new e2e or CI coverage
- benchmark or metrics documentation
- non-breaking config guide updates
- install extras and Docker images
- contributor or release tooling improvements

Example phrasing:

- "Expanded multi-node training documentation and added e2e coverage for offline DPO on SD3.5."

### Breaking Changes

Purpose: incompatible changes requiring migration.

Typical content:

- Hydra config renames or removals
- CLI or pipeline API signature changes
- default backend or behavior changes

## Inclusion Heuristics

Include:

- new verified model x algorithm recipes under `Model & Algorithm Supports`
- rollout or trainer architecture changes users will notice
- meaningful training throughput, stability, or determinism improvements with a concrete setup
- reward pipeline changes that affect training workflows
- Ascend NPU or other hardware backend support
- explicit upgrade caveats or compatibility warnings in the relevant content section

Usually merge or omit:

- pure typo or formatting docs cleanup
- lint-only PRs
- internal CI maintenance with no release-facing impact
- repeated follow-up fixes already captured by one stronger summary
- WIP pipelines presented as fully verified support

## Tone and Density

- README `News` bullets are good compact examples for Highlights
- `verl` final releases are good references for Breaking Changes clarity
- Keep bullet density moderate; a reader should scan one section in under a minute
- Prefer 4-5 filled body sections over padding empty headings
- Do not include a release-note change claim, caveat, compatibility note, or recipe-status note unless it has a corresponding PR citation; move uncited material to the review file
- Avoid alternating prose with many inline commands, paths, package names, or config keys; group exact commands and version pins into one compact install/compatibility bullet when needed

## Common Mistakes

| Mistake | Better approach |
|---------|-----------------|
| One bullet per PR | Merge PRs into a release-facing narrative |
| Listing backend refactors without impact | Explain what training or rollout capability is unlocked |
| Mixing breaking config changes into feature sections | Move them to `Breaking Changes` |
| Letting `What's Changed` drive the whole note | Use it as source material, not final structure |
| Claiming Planned/WIP support as verified | Match README status and example readiness |
| Hiding trainer changes under rollout | Put trainer-engine work under `Architecture / Training`, and rollout engine work under `Architecture / Rollout Engine` |
| Adding a separate non-breaking caveats section | Fold non-breaking caveats into `Documentation / Tooling`, `Hardware`, or the relevant feature section |
| Including an uncited caveat or status note in the release body | Move it to `4-release-note-review.md` unless a PR citation can be attached |

## Baseline Example Derived from README News

The following is a cold-start example of how early release notes may read. Replace with real tag-based content once releases exist.

```md
## Highlights

VeRL-Omni `v0.1.0` is the first focused release for multimodal generative RL training, covering diffusion, unified understand-and-generate, and omni-modality model families.

### Architecture

#### Rollout Engine

* Upgraded vLLM-Omni rollout backend to v0.22 for higher throughput.
* Optimized rollout batching and LoRA weight updates.

#### Training

* Switched default actor attention backend to FA3.
* Integrated Flow-DPPO training path for Qwen-Image.

#### Reward

* Added async reward and HTTP scorer support for flexible reward serving.

### Model & Algorithm Supports

* Added verified DiffusionNFT and Diffusion DPO recipes for Qwen-Image and SD3.5, with docs and example trainers.
* Introduced Qwen3-Omni GSPO trainer and Wan2.2 DanceGRPO video generation support.

### Hardware

* Added Ascend NPU support with FlowGRPO quickstart and e2e coverage.

### Documentation / Tooling

* Refreshed installation, quickstart, and e2e validation docs for first-release recipes, including the dependency refresh required by the vLLM-Omni pin update.
```
