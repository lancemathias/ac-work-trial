# Experiment 5: Residual MLP readout

Human-directed, October 15, 2026. The fixed-order residual MLP is the completed Experiment 5 baseline. The original per-epoch-shuffled branch was terminated and is superseded by [Experiment 6: Option orders](experiment-6-option-orders.md), which is only the v3.1 cached-permutation run.

Current recipe and accepted selection: [W&B config](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/c09mctl9/overview). The original paired protocol below is retained as historical provenance; `fixed/*` and `shuffled/*` share one W&B run.

Both use the frozen Qwen/Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The final prompt-token feature
feeds LayerNorm, Linear(5120, 1024), GELU, Linear(1024, 255). Add these
corrections to the frozen LM output scores for the 255 verified code tokens.
Mask slots beyond the current question's option count before cross entropy
or softmax. Trainable parameters: 5,515,519. The backbone and selected output
rows remain frozen; no adapters.

Initialization is **Xavier uniform (Glorot uniform), gain 1.0**, independently
applied to each linear weight matrix. Entries are sampled from the symmetric
zero-mean distribution U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))].
Both linear biases are zero. LayerNorm scale is one and bias zero. The two
conditions start with identical sampled parameters and use the same record
minibatches. The MLP output matrix is random, not zero-initialized.

Reuse the original v3 release (7,928 records, SHA256
`ef15c46f49192cafe5d35a1b0567e1a33cc13807fb5c2ca4d50e1fdbc6beec81`)
and its existing grouped train/dev/calibration splits. Train for five epochs
with batch 64, AdamW learning rate 3e-4, matrix-only weight decay 0.01,
10% linear warmup, cosine decay, gradient clip 1.0, seed 20261014. Both
conditions select the earliest checkpoint with minimum native-order dev NLL.
Fit a separate scalar temperature for each selected head on calibration only.

Shuffling is derived from SHA256(seed, record ID, epoch); move each option's
ID and meaning together, reassign codes, and remap the target by semantic ID.
Keep evidence and question unchanged. Frozen features must be re-extracted
for each distinct prompt permutation. Cache final-position vectors in
checksummed resumable shards, sharing native training and evaluation features.
An epoch snapshot contains head and optimizer state.

Evaluate both selected heads on the existing sampled benchmark panel in native,
reverse, and one seeded shuffled order. Align distributions to semantic IDs
before scoring. Report accuracy, NLL, Brier, calibration, argmax flip rate,
and total variation across orders. Benchmark labels never enter prompts,
checkpoint selection, or temperature fitting. This is one paired seed, not
a multi-seed estimate of uncertainty. The 255-code interface is verified;
training on v3 does not establish performance at every possible option count.

The host runner waits for existing compute to release before requesting one
A100-80GB in workspace `ac-worktrials`, profile `ac-worktrial`, environment
`worktrial-lance`. It preserves the original reservation cutoff and exports
all results and cached tensors locally with checksum verification. Persistent
Modal volumes are retained. It does not stop or modify other experiments.

Configuration: `configs/readout-mlp/experiment.json`. Launch only after
`modal_residual_readout.py::validate` passes for the exact uploaded source.
