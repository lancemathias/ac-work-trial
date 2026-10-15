# Descriptive autoresearch resource names

The partial bidirectional-attention experiment is named **Bidirectional full attention**; its canonical run name is `attention-bidir-qwen38-27b-v3-seed20261014`. The combined readout is **Joint-head + native mixture**, canonical run name `cached-joint-head-readout-equal-mixture-v1`. Both retain their `autoresearch` classification.

The [resource alias catalog](../configs/resource-aliases.json) maps descriptive names to immutable legacy identifiers. Original W&B run IDs and URLs, checkpoint bytes and hashes, execution receipts, volume paths and Git history remain valid. An old experiment-number token inside a legacy path is provenance, not its current display label.

## W&B

- [Bidirectional full attention](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/dyjus5wh)
- [Joint-head + native mixture](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/6spap1d0)
- [Retrospective attention evaluation](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist4d6cf18546e78e893ffd)
- [Retrospective mixture evaluation](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/histbb478856e84c09dc39d4)

The descriptive evidence collection is `lmathias/ac-worktrial-decisions/attention-bidir-qwen38-27b-v3-seed20261014`. It references every file in the original evidence artifact and includes the alias catalog; no model is retrained or tensor rewritten. See the [W&B verification receipt](../outputs/handoff/wandb-resource-renames.json).

## Modal

Workspace `ac-worktrials`, profile `ac-worktrial`, environment `worktrial-lance`. The alias catalog is published in volume `ac-worktrial-training-results` at `resource-aliases/attention-bidir.json`. Its [verification receipt](../outputs/handoff/modal-resource-aliases.json) records all eight existing directories checked.

These are application-level aliases, not filesystem moves or symlinks. Resolve a `canonical_alias` to its `legacy_path` and use that path in the listed volume. This preserves every previously saved checkpoint and cache. Only the accepted checkpoint is promoted in the [checkpoint catalog](accepted-checkpoints.md).

The renamed launcher, [modal_attention_bidir.py](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/scripts/modal_attention_bidir.py), uses app name `ac-worktrial-autoresearch-attention-bidir` on a future invocation. No app is launched or restarted by this naming change.

## GitHub

The descriptive source files are [attention configuration](../configs/attention-bidir-qwen38-27b-v3.json), [cache source](../configs/attention-bidir-cache-source.json), [launcher](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/scripts/modal_attention_bidir.py) and [attention results](autoresearch-attention-bidir-results.md). The cache-source config embeds the original immutable training-source identity for replay checks; its legacy labels remain intentionally unchanged.

The branch alias `autoresearch/attention-bidirectional` points to original experiment commit `d9cefcbe1b2043667da3b6ecaaa291ccaabdcfb6`. The original branch ref remains reachable. Current source/report changes are on `deliverables/research-handoff` and `deliverables/accepted-model`; `main` is updated through the user-created PR.

The [full change inventory](resource-naming-change.md) records the original scope. [Execution verification](../outputs/handoff/resource-naming-verification.json) checks unchanged scientific scores, historical bytes and accepted-checkpoint references.
