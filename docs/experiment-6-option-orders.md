# Experiment 6: Option orders

Human-directed, October 15, 2026. Scope: only the v3.1 cached-permutation run. The original per-epoch-shuffled attempt under [Experiment 5](experiment-5-residual-mlp.md) was terminated and superseded by this run.

[W&B config](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/28lh244a/overview).

This experiment uses one cached shuffled view of each Choice example, combined
with the cached v3 data, followed by a matched training run. The interrupted
paired experiment already contained the necessary first-permutation features,
so no backbone extraction was repeated.

This is a feature-view release, `frozen-readout-feature-views-v1`, rather than
new semantic source examples. Training contains 6,262 native views plus
5,493 choice-only permutation views, totaling 11,755. Exactly the first seeded
permutation is retained, including 973 chance identity permutations. Each
augmented view has a unique view ID and preserves the original semantic ID,
family/group, option meanings, and correct semantic target. No noul views are
added. Choice examples therefore receive twice the view sampling weight of
noul examples. Validation (809) and calibration (857) remain unchanged.
The evaluation-only 2,111-example native benchmark cache is referenced separately.

The dataset is `readout-feature-dataset-v3.1-choice-permutation1` on the
`ac-worktrial-training-results` Volume in `worktrial-lance`. Its manifest binds
the original dataset and split hashes, source checksums, code mapping, backbone
revision, permutation seed/epoch, and cached feature inventories. CPU tests check
choice filtering, target remapping, stable family membership, unique view IDs,
and identity permutations.

The run is `readout1024-qwen38-v3.1-choice-permutation1-seed20261014`. It starts
from the exact same saved initial head as the v3 fixed baseline: Xavier uniform
MLP weights, zero biases, 1,024 hidden units, frozen answer-code output rows.
It uses the same five epochs, batch 64, AdamW learning rate 3e-4, matrix-only
decay 0.01, 10% warmup, cosine decay, clip 1.0, seed 20261014, native dev
accuracy selection with earliest ties, and separate calibration temperature.
All features, including calibration and test, are cached; the backbone is never
loaded. Five epochs now mean 920 optimizer updates versus the baseline's 490.
This contrast consequently changes both augmentation and total updates.

The v3 fixed baseline completed training and native benchmark inference, then
failed during report generation because its cached predictions had no latency
measurement. CPU recovery fixes optional latency reporting (unknown is null),
revalidates predictions, publishes the model artifact in W&B run `c09mctl9`, and
packages all parent/fixed tensors, checkpoints, optimizer states, and traces.
Recovered macro benchmark accuracy: 70.4180643%. The original W&B notes explain
the shared fixed/shuffled logging counter and fixed-only resume.

CPU bundle jobs export standalone archives and file SHA256 manifests locally.
Persistent Modal volumes are preserved. The v3.1 model receives its own W&B run.
