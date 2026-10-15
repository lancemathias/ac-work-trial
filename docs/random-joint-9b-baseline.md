# Qwen3.5-9B random joint-head baseline

Requested matched control for the Qwen3.8-27B random joint-head run. The original
pretrained backbone is frozen, not randomly initialized. Three untrained CLEF
joint heads use seeds 20261014, 20261015 and 20261016. Report the arithmetic mean
and sample standard deviation of per-seed scores, not a probability ensemble.

Checkpoint: `Qwen/Qwen3.5-9B` at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The head input dimension changes
from 5120 to 4096 to match the checkpoint; width 1024, two routing layers, four
transformer layers, 16 heads and feedforward dimension 4096 remain unchanged.
Retain the native CLEF lexical prior and frozen Qwen output embeddings, BF16 backbone,
FP32 head with CUDA BF16 autocast as in E1,
batch size one, no truncation, 8192-token ceiling, T=1 and no training or fitting.
Use the identical frozen 2111-example panel and BBH subtask-macro scoring.
One backbone forward is shared across heads for each example, as in the 27B run.
The one-hour GPU cap includes loading; the runner stops softly at 3500 seconds.

```sh
.venv/bin/modal run --profile ac-worktrial --env worktrial-lance scripts/modal_random_joint_e1.py
```

Use workspace `ac-worktrials`, profile `ac-worktrial`, environment
`worktrial-lance`. Export configs, predictions, metrics and initial head tensors
with checksums locally before cleaning resources. Raw benchmark data and the
separate human audit sample stay outside Git. This run has no model training.

The first seed loads E1's actual saved pre-training head from
`ac-worktrial-training-results/e1-qwen35-9b-seed20261014-6f89ea4/initial-head.safetensors`,
SHA-256 `2fe908f8ac0f25f81238b4f4c59b23377dc27665532741483b043a1b99bcf957`.
The result volume is mounted read-only. Seeds 20261015 and 20261016 are fresh
initializations of the same architecture. This makes the first seed a direct
before-training control for E1, while the three-seed average measures
initialization variability. Match E1 Torch 2.9.1 and torchvision 0.24.1; other
model dependencies retain their pinned versions. Output directory:
`outputs/baselines-20261014/random-joint-qwen35-9b-e1-matched`.

A preliminary Torch 2.14 / BF16-head run, app `ap-L3D6y0P3dwLjGGoW7Ngqqs`,
was stopped after the user requested exact E1 matching. It is excluded from
reported results. Its partial outputs are preserved separately on Modal under
`random-joint-qwen35-9b`; no results are mixed into the replacement run.

## Completed result

All three seeds attempted all 2111 examples. Each rejected the same single
ContractNLI input above the 8192-token ceiling; all other predictions were valid.
The 11-suite macro is **39.92% ± 3.21 pp valid-only**, or **39.90% ± 3.21 pp
counting the miss as wrong** (sample standard deviation across seeds).
Coverage-adjusted seed macros are 42.95%, 40.20% and 36.54% in seed order.
The actual E1 pre-training checkpoint (first seed) scores 42.95%, compared with
57.35% for the selected trained E1 checkpoint on the same coverage-adjusted
metric: a 14.40 percentage-point increase. The three-seed untrained mean is not
a matched three-seed training comparison; only one trained E1 seed exists here.

The GPU worker completed in 489.48 seconds (8.16 minutes), with 20.84 GiB peak
allocated memory and unchanged backbone parameter version counters. Source
checkpoint, configs, traces and metrics are recorded in the run directory.
The separate preliminary run is excluded. The main baseline results table
contains per-benchmark seed means and sample standard deviations.
