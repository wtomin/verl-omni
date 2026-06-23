---
name: verl-omni-release-note-writer
description: Use when drafting or editing release notes for verl-project/verl-omni, especially when summarizing changes between tags, organizing highlights around algorithms/recipes/backends, and matching VeRL-Omni release style
---

# VeRL-Omni Release Note Writer

## Overview

This skill writes release notes for `verl-project/verl-omni` by following an editorial workflow adapted from the VeRL-Omni roadmap style. Organize the body around **Architecture** (`Rollout Engine`, `Training`) and **Model & Algorithm Supports**, with optional sections for documentation, hardware, upgrade caveats, and breaking changes.

Always read [references/past-release-note-patterns.md](references/past-release-note-patterns.md) first, then use [references/release-note-template.md](references/release-note-template.md) as the drafting template.

## When to Use

- Drafting a new VeRL-Omni release note from merged PRs or a GitHub compare view
- Rewriting an auto-generated `What's Changed` dump into a user-facing summary
- Editing an RC or final release note before the first or nth GitHub Release
- Cross-checking whether a change belongs in a themed section, `Breaking Changes`, or should be omitted

Do not use this skill for `verl`, `vllm-omni`, or unrelated repositories.

## Output Workspace

Save working files under `verl-omni-release-note/output/$VERSION/`.

Recommended files:

- `0-raw-input.md`: compare output, PR list, and rough notes
- `1-commit-triage.csv`: per-PR inclusion and category decisions
- `2-highlights-draft.md`: short editorial summary
- `3-release-note-draft.md`: full release note draft
- `4-release-note-review.md`: questions, uncertainties, and follow-up checks

## Core Workflow

### 1. Gather the release boundary

Identify:

- current tag, for example `v0.1.0`
- previous tag
- whether the target is an RC or final release

Tag selection rules:

- For a final release, `current tag` is the tag of the release being written, and `previous tag` is the previous final release tag.
- For an RC release, `current tag` is the tag of the RC being written, and `previous tag` is the immediately previous final or RC release tag in the release chain.
- For the first release, set `previous tag` to `initial` and state in the opening paragraph that this is the first tagged release.

Use one or more of:

- `https://github.com/verl-project/verl-omni/releases`
- `https://api.github.com/repos/verl-project/verl-omni/releases`
- `https://api.github.com/repos/verl-project/verl-omni/compare/...`
- README `News` section and `examples/*/README.md` as supplementary sources

If a release body already exists, treat it as one source, not ground truth. Re-check important claims against PRs and verified recipes when wording matters.

### 2. Build a triage sheet

Review each PR or commit and record:

- title
- PR number
- PR module tag from title, for example `[diffusion, rollout]`
- model family and algorithm, if applicable
- user-facing summary
- section
- decision: `include`, `merge-into-summary`, `ignore`
- reason

Ignore or merge away low-signal items such as:

- typo-only docs with no recipe impact
- lint-only or formatting-only PRs
- trivial CI maintenance with no release-facing impact
- internal cleanup with no user-visible outcome
- duplicate fixes already covered by a stronger umbrella PR

### 3. Convert engineering changes into release language

Write for users running recipes, not for the merge log.

Prefer:

- "Added verified DiffusionNFT recipes for Qwen-Image and SD3.5"
- "Upgraded vLLM-Omni rollout backend to v0.22 with higher throughput"
- "Integrated Flow-DPPO for Qwen-Image with example scripts under `examples/flowdppo_trainer/`"

Avoid:

- PR-title fragments
- implementation-only detail with no user impact
- one bullet per PR when several PRs clearly belong to one theme
- claiming support for WIP or Planned recipes as fully verified

### 4. Write the Highlights block first

The opening should do three things:

1. state release scale or context when useful
2. summarize the main story of the release in 1 paragraph
3. list 4-8 key improvements that a user should scan first

Recent VeRL-Omni communication consistently leads with:

- new **algorithm x model x recipe** combinations
- rollout or trainer architecture upgrades (vLLM-Omni, FSDP2, VeOmni)
- reward pipeline or async scoring improvements when they affect architecture
- throughput, stability, or determinism gains with a concrete anchor
- Ascend NPU or other hardware backend coverage
- dependency or install changes users must act on

RC releases use the same template as final releases. Keep the `rc` version identifier in the title and opening copy, but do not switch to a separate RC-only structure.

### 5. Expand into the VeRL-Omni roadmap directory

Use the section bank below. Prefer the VeRL-Omni roadmap directory:

1. `Architecture`
   1. `Rollout Engine`
   2. `Training`
   3. `Reward`
2. `Model & Algorithm Supports`

Add optional sections such as `Documentation / Tooling`, `Hardware`, and `Breaking Changes` only when the release has substantive content that does not fit the primary directory.

Rules:

- omit empty sections
- group related PRs into one paragraph or bullet cluster
- mention representative PR numbers, not every PR, and keep PR citations bolded as `**(#1234)**`
- link to docs or `examples/` when a recipe is user-facing
- preserve important caveats, flags, and compatibility notes in the most relevant content section or `Breaking Changes`

### 6. Add caveats and breaking changes explicitly

If a release includes:

- Hydra YAML or CLI breaking changes
- manual dependency bumps or reinstall steps
- hardware or backend limitations
- experimental or WIP support
- incompatible default changes

surface them in `### Breaking Changes` or the most relevant content section. Do not create a separate non-breaking caveats section.

## Section Bank

Use these body sections plus `Breaking Changes`. Omit empty sections. Do not create a separate non-breaking caveats section. Use `####` subsections when a section has multiple independent themes.

### Architecture

Rollout, trainer, reward-loop, and training-system changes.

Recommended subsections:

- `#### Rollout Engine` for vLLM-Omni version alignment, async rollout, adapters, rollout servers, batching, rollout weight updates, rollout correction, and rollout-side performance
- `#### Training` for FSDP2, VeOmni, diffusers trainer engine, reward-loop integration, sequence parallelism, determinism, MFU, memory, startup, or training throughput
- `#### Reward` for reward managers, weighted aggregation, HTTP scorer integration, external reward services, async reward overlap, and reward-loop correctness or latency

Include:

- vLLM-Omni rollout backend changes, async rollout, rollout adapters, rollout servers
- rollout batching, LoRA weight update, deterministic rollout behavior, rollout correction, rollout throughput or latency
- FSDP2, VeOmni, diffusers training adapters, USP/TP/DP, and training-side performance
- reward managers, HTTP scorer, async reward overlap, and scorer services
- reproducibility, stability, memory, startup, and trainer metrics such as diffusion MFU

Do not include:

- new algorithms, models, or runnable recipe claims unless the architectural change is inseparable from the recipe
- platform-only Ascend NPU support unless it changes the generic rollout/trainer architecture
- docs-only edits unless they explain a new architecture capability

### Model & Algorithm Supports

New or updated algorithms, model families, pipelines, and runnable recipes.

Recommended grouping:

- image generation / X2I, for example Qwen-Image, SD3.5, HunyuanImage, GLM-Image
- video generation / X2V, for example Wan2.1/Wan2.2
- audio generation / X2A, for example PrismAudio
- unified understanding and generation, for example BAGEL
- omni models, for example Qwen3-Omni

Include:

- new algorithm integrations (FlowGRPO, Flow-DPPO, GSPO, DPO, DiffusionNFT, DanceGRPO, MixGRPO, GRPO-Guard, SRPO, etc.)
- new model x algorithm pipelines under `verl_omni/pipelines/`
- new or updated `examples/` and docs pages tied to runnable recipes
- recipe status changes, especially WIP or Planned support becoming verified

Do not include:

- rollout backend internals unless they are inseparable from the recipe
- trainer engine work that applies across many recipes
- pure docs, tests, or install changes

### Hardware

Platform-specific execution backends and hardware-facing training support.

Include:

- Ascend NPU support and NPU quickstart/e2e coverage
- multi-node training support that is hardware- or cluster-facing
- backend-specific runtime constraints users must know on non-GPU platforms

Do not include:

- generic FSDP/VeOmni changes that apply equally on GPU
- docs-only edits unless they are NPU-specific guides

### Documentation / Tooling

Remaining user-facing docs, tests, CI, install, and release tooling.

Include:

- documentation refreshes, new guides, API reference updates
- new or expanded e2e tests and CI workflow coverage
- non-breaking config documentation, developer guides, benchmark infra
- contributor workflow, Docker, install extras, or release pipeline improvements

Do not include:

- breaking config or API changes (use `Breaking Changes`)
- substantive recipe, rollout, reward, or hardware features (move to the relevant section above)

### Breaking Changes

Hydra config renames, CLI signature changes, removed pipelines, default behavior changes, and migration steps.

## Section Selection Rules

Use this mapping as the default:

| Change type | Preferred section |
|-------------|-------------------|
| New algorithm, new model x algo recipe, new example | Model & Algorithm Supports |
| PR modules: `algo`, `diffusion`, `omni`, `model`, `data` for recipe-facing work | Model & Algorithm Supports |
| vLLM-Omni rollout, async server, rollout adapter, rollout batching, rollout correction, rollout weight updates | Architecture / Rollout Engine |
| PR modules: `rollout`, `vllm_omni` | Architecture / Rollout Engine unless clearly trainer-only |
| FSDP2, VeOmni, Ray worker, USP/TP/DP, deterministic trainer, training MFU | Architecture / Training |
| PR modules: `trainer`, `fsdp`, `ray`, `worker` | Architecture / Training unless clearly recipe-only |
| Reward manager, HTTP scorer, async reward | Architecture / Reward |
| PR module: `reward` | Architecture / Reward unless the reward is recipe-specific |
| Ascend NPU, NPU e2e, hardware-specific runtime | Hardware |
| Docs, benchmarks, CI, e2e scripts, non-breaking cfg docs | Documentation / Tooling |
| PR modules: `doc`, `ci`, `tests`, `cfg`, `ckpt`, `docker`, `misc` | Documentation / Tooling unless clearly training/rollout/reward/hardware work |
| Dependency reinstall, experimental support, soft caveats | Documentation / Tooling, Hardware, or the most relevant content section |
| Breaking API/config/default changes | Breaking Changes |

Performance claims:

- rollout throughput or latency: **Architecture / Rollout Engine**
- training MFU or end-to-end training throughput: **Architecture / Training**
- reference benchmark against external baselines: mention the anchor model/setup in the claim

## Writing Rules

- Prefer one paragraph plus grouped bullets over a long flat bullet list
- Mention PR numbers in bold `**(#1234)**` form after the claim
- Do not include a release-note change claim, caveat, compatibility note, or recipe-status note unless it has a corresponding PR citation. If the source is README/docs only or maintainer memory without a PR number, move it to `4-release-note-review.md` instead of `3-release-note-draft.md`.
- Pair algorithm names with model families and example paths when verified
- Keep claims specific and user-facing
- If several fixes land for one recipe, merge them into one sentence
- Do not overstate internal refactors; explain what they unlock for training or rollout
- Avoid alternating prose with many inline commands, paths, package names, or config keys. Group install commands, version pins, paths, and compatibility details into one compact paragraph or bullet instead of scattering code-formatted fragments through every sentence.
- Prefer readable prose over dense code-formatted text. Use code formatting only for exact paths, commands, package names, config keys, and model IDs that users may copy or search.
- Keep `What's Changed` and `New Contributors` as GitHub-generated appendices if they already exist
- Sync major items with README `News` or note the intentional difference in `4-release-note-review.md`

## Validation Checklist

- [ ] RC and final releases both use the Final Release Template structure
- [ ] `current tag` is the tag of the release being written
- [ ] `previous tag` follows the release-boundary rule
- [ ] Opening paragraph states the release story clearly
- [ ] Highlights contain only the most important user-facing items
- [ ] Section headings match the VeRL-Omni roadmap directory; empty sections are removed
- [ ] Verified recipes are distinguished from WIP or Planned support
- [ ] Pure maintenance noise is omitted or merged away
- [ ] Important caveats, dependency actions, and breakages are explicit
- [ ] PR numbers, model names, and example links are accurate
- [ ] Every release-note body bullet that describes a change or caveat has at least one PR citation
- [ ] Any uncertain claim is marked in `4-release-note-review.md`

## Research Tips

- For ambiguous PRs, inspect the PR body via `https://api.github.com/repos/verl-project/verl-omni/pulls/<number>`
- Use the compare API to avoid missing merged work between tags
- Cross-check recipe status against README model/algorithm table and `examples/*/README.md`
- Check `tests/special_e2e/` for e2e coverage that supports a release claim
- If a release reuses content from an RC, make the delta versus the prior RC explicit
- When a refactor is large, summarize the user-visible consequence instead of enumerating files

## References

- [past-release-note-patterns.md](references/past-release-note-patterns.md) - Baseline style, cold-start patterns, and section definitions
- [release-note-template.md](references/release-note-template.md) - Copyable release note template for RC and final releases
