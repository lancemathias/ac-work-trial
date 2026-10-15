# Qwen3.5-9B random joint-head baseline

Requested matched control for the Qwen3.8-27B random joint-head run. The original
pretrained backbone is frozen, not randomly initialized. Three untrained CLEF
joint heads use seeds 20261014, 20261015 and 20261016. Report the arithmetic mean
and sample standard deviation of per-seed scores, not a probability ensemble.

Checkpoint: `Qwen/Qwen3.5-9B` at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The head input dimension changes
from 5120 to 4096 to match the checkpoint; width 1024, two routing layers, four
transformer layers, 16 heads and feedforward dimension 4096 remain unchanged.
Retain the native CLEF lexical prior and frozen Qwen output embeddings, BF16,
batch size one, no truncation, 8192-token ceiling, T=1 and no training or fitting.
Use the identical frozen 2111-example panel and BBH subtask-macro scoring.
One backbone forward is shared across heads for each example, as in the 27B run.
The one-hour GPU cap includes loading; the runner stops softly at 3500 seconds.

```sh
.venv/bin/modal run --profile ac-worktrial --env worktrial-lance scripts/modal_random_joint.py --variant qwen35-9b
```

Use workspace `ac-worktrials`, profile `ac-worktrial`, environment
`worktrial-lance`. Export configs, predictions, metrics and initial head tensors
with checksums locally before cleaning resources. Raw benchmark data and the
separate human audit sample stay outside Git. This run has no model training.
