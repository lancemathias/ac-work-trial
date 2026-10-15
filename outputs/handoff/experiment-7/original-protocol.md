# PPLX recipe: 9B full-weight, 10k rows, two epochs

User instruction 2026-10-15 revises the planned 27B replication: uniformly sample
the reconstructed corpus to 10,000 training rows, use an unfrozen 9B backbone,
and train two epochs with the released parameters. This explicitly authorizes
backbone and vision-encoder updates for this run, superseding the earlier
frozen-only research constraint within this experiment.

Base: Qwen/Qwen3.5-9B at c202236235762e1c871ad0ccb60c8ee5ba337b9a.
Upstream PPLX source: d8c3bd18e31dc1804433d38528b3e770019bf00c. Its training
modules and lockfile match the historical training-code hashes in the release.
The exact curated 73k corpus is unpublished; the published builder produced
96,000 rows. Python Random(20260920).sample chooses exactly 10,000 without
replacement, including 99 generated-image examples. No source is silently
removed. Development (1,497) and calibration (983) retain their original folds.
Fresh Jev1.13 predictions on development supply the upstream calibration
eligibility reference; the original reference file is not published. The fixed
2,111-example benchmark panel is excluded from training/calibration/selection.

Model: the original 255-output bias-free linear decision readout initialized
from pretrained answer-code LM-head rows. No added nonlinear head or adapters.
All retained model weights are trainable. Upstream CPUOffloadAdamW retains FP32
masters and moments. Learning rate 2e-6, decay 0.01 on all upstream parameters,
batch limit 32, effective batch 256, token budget and max length 8192, gradient
clip 1.0, CPU threads 32, seed 20260920. The same 5% warmup/cosine schedule ends
at 10% of peak LR. Two epochs produce 80 updates (40 per epoch). Original option
augmentation, soft/hard targets, calibration and checkpoint eligibility/selection
are retained. Monitor/evaluate/save-resume every 50 updates plus the final step.

The locked training environment is installed with uv --frozen because the
released lock contains cutoff metadata that makes uv --locked attempt to
re-resolve it. No lock entries or upstream numerical code are changed. W&B
runs in the outer Modal environment and reads subprocess JSON events, so its
extra dependencies do not modify the upstream locked numerical environment.
The copied upstream source receives a standalone git snapshot for its required
runtime provenance; upstream and adapter hashes are recorded separately.

A single A100-80GB with 32 CPU cores and 320GiB host RAM runs training. The
terminal model and the upstream-selected checkpoint are both evaluated on the
11-suite panel after training. If the selected checkpoint is the terminal
update, identical test predictions are reused. Source data, sample IDs, hashes,
events, optimizer state and model checkpoints are preserved. No GPU watchdog or
new recurring automation is created. Modal workspace ac-worktrials, profile
ac-worktrial, environment worktrial-lance.
