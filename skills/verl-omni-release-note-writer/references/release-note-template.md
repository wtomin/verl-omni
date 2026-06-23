# VeRL-Omni Release Note Templates

Use this template after triaging PRs and drafting the main themes.

Use the same structure for both RC and final releases. For RCs, keep the `rc` suffix in `<VERSION>` and make the opening paragraph explicitly say it is a release candidate.

Before drafting, set the comparison boundary correctly:

- Final release: `current tag = this final release tag`, `previous tag = previous final release tag`
- RC release: `current tag = this RC tag`, `previous tag = immediately previous final or RC release tag`
- First release: `previous tag = initial`

## Final Release Template

```md
## Highlights

This release includes <N> merged PRs from <contributors> contributors<, including <new contributors> new contributors>.

VeRL-Omni `<VERSION>` <release story in one sentence>. It expands <algorithm/model/recipe scope>, improves <architecture story across rollout engine and trainer>, and updates <hardware/docs/tooling story>.

If `<VERSION>` is an RC, add one sentence such as: `This release candidate is intended to validate <focus area> before the final cut.`

If this is the first tagged release, add one sentence such as: `This is the first tagged release since multimodal generative RL training moved into a dedicated VeRL-Omni repository.`

### Architecture

#### Rollout Engine

* <Grouped summary of vLLM-Omni rollout, async servers, rollout adapters, batching, rollout correction, rollout weight updates, and rollout-side performance>. **(#1234, #2345)**

#### Training

* <Grouped summary of FSDP2/VeOmni, diffusers trainer engines, parallelism, determinism, and training-side performance>. **(#1234, #2345)**

#### Reward

* <Grouped summary of reward managers, scorer integrations, HTTP scorer, async reward overlap, and reward-loop correctness or latency improvements>. **(#1234, #2345)**

### Model & Algorithm Supports

* <Grouped summary of new algorithms, model families, pipelines, and verified recipes. Group by X2I, X2V, X2A, unified understanding/generation, or omni models when useful>. **(#1234, #2345)**

### Hardware

* <Grouped summary of Ascend NPU, multi-node hardware-facing support, and platform caveats>. **(#1234, #2345)**

### Documentation / Tooling

* <Grouped summary of docs, install extras, Docker, e2e tests, CI coverage, benchmark infra, and non-breaking config guides>. **(#1234, #2345)**

## Breaking Changes

* <Action-oriented Hydra config rename / CLI change / removed pipeline / incompatible default change with migration step>. **(#1234)**
```

## Section Omission Rules

- Remove any body section or subsection that has no substantive content
- Do not create alternate top-level headings for RC releases
- Move install pin changes to `Documentation / Tooling` unless they are part of a breaking migration
- Move Hydra/CLI/API breakages to `Breaking Changes`, not `Documentation / Tooling`

## Editing Rules

- Keep the opening paragraph to 2-3 sentences
- Prefer grouped summaries over long bullet dumps
- Keep PR citations representative rather than exhaustive
- Keep PR citations bolded as `**(#1234)**`
- Do not include a release-note change claim, caveat, compatibility note, or recipe-status note unless it has a corresponding PR citation; move uncited material to the review file
- Link to `examples/` or docs pages for every major recipe claim
- Distinguish verified recipes from WIP or Planned support
- Do not create a separate non-breaking caveats section
- Avoid alternating prose with many inline commands, paths, package names, or config keys; group exact commands and version pins into one compact install/compatibility bullet when needed
- Do not create a separate RC-only section layout; RCs use this same template

## Triage CSV Columns

Use these columns in `1-commit-triage.csv`:

```csv
pr_number,title,pr_modules,model,algorithm,section,decision,user_summary,reason
1234,"[diffusion, algo] feat: add DiffusionNFT","diffusion,algo",Qwen-Image,DiffusionNFT,"Model & Algorithm Supports",include,"Added verified DiffusionNFT recipe for Qwen-Image","New user-facing recipe"
5678,"[rollout, vllm_omni] feat: bump vllm-omni","rollout,vllm_omni",,,"Architecture / Rollout Engine",include,"Upgraded vLLM-Omni rollout backend to v0.22","Backend upgrade affects install and throughput"
```

Valid `section` values:

- `Architecture / Rollout Engine`
- `Architecture / Training`
- `Architecture / Reward`
- `Model & Algorithm Supports`
- `Hardware`
- `Documentation / Tooling`
- `Breaking Changes`
- `ignore`
