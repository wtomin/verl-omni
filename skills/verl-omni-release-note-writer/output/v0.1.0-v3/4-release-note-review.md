# VeRL-Omni v0.1.0-v3 Release Note Review

## Validation Checklist

- [x] Final release template structure is used.
- [x] `current tag` is treated as `v0.1.0`.
- [x] `previous tag` is treated as `initial`.
- [x] Opening paragraph states the first-release story.
- [x] Main body uses the requested directory: `Architecture`, `Architecture / Rollout Engine`, `Architecture / Training`, `Architecture / Reward`, and `Model & Algorithm Supports`.
- [x] Reward-related PRs are listed under `Architecture / Reward`, parallel to `Rollout Engine` and `Training`.
- [x] Experimental rollout correction is classified under `Architecture / Rollout Engine`.
- [x] Hardware section is named `Hardware`.
- [x] No separate non-breaking caveats section is used.
- [x] Install and compatibility reminders are folded into `Documentation / Tooling`.
- [x] WIP and Planned recipe caveats without PR citations are kept out of the release body.
- [x] Text avoids frequent prose/code alternation by grouping install and compatibility details.
- [x] Representative PR numbers are included in bold `**(#1234)**` form.
- [x] Every release-note body bullet that describes a change or caveat has at least one PR citation.
- [x] Breaking rollout client adaptation is called out separately.

## Open Questions Before Publishing

- Confirm whether the public release tag should be `v0.1.0` or another version string before publishing.
- Re-run the final compare after the release tag is cut; the v3 draft inherits the existing count of 133 merged PRs from 14 contributors.
- Confirm whether PRs merged after the previous source collection window should be included.
- Confirm the exact migration steps for `[BREAKING][rollout]` `#52`.
- Confirm whether GitHub-generated `What's Changed` and `New Contributors` appendices should be appended after this curated body.

## Claims To Re-Check Against Maintainer Intent

- The draft describes `v0.1.0` as the first tagged release for the dedicated VeRL-Omni repository.
- LTX2.3 FlowGRPO and HunyuanImage-3.0 MixGRPO/SRPO are intentionally described as WIP or Planned, not verified recipes.
- Async reward is cited through `#155` as user-facing documentation; implementation ownership may span additional PRs.
- The README throughput claim is not repeated in the body because the exact benchmark citation should be confirmed before publishing.
- Multi-node and 64-card examples are treated as documentation/tooling, not as a standalone performance guarantee.

## Suggested Publish Steps

1. Cut or fetch the final `v0.1.0` tag.
2. Re-run compare/release API collection against the final boundary.
3. Refresh PR count, contributor count, and new contributor count.
4. Verify representative PR citations in `3-release-note-draft.md`.
5. Confirm the Breaking Changes wording for `#52`.
6. Paste `3-release-note-draft.md` into the GitHub Release body and append GitHub-generated sections if desired.
