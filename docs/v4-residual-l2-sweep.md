# Residual L2 sweep on v4

User-requested E9 protocol rerun on the expanded v4 frozen-feature dataset, with lambda `0.03, 0.1, 0.3, 1, 3`, ten epochs each. The protocol is specified in `configs/v4-residual-l2-sweep.json`. Runs publish metrics and configuration in the [W&B group](https://wandb.ai/lmathias/ac-worktrial-decisions/groups/v4-residual-l2-sweep); checkpoints are exported locally rather than uploaded to W&B.

Every lambda starts from the exact E9 initial head (E8 saved zero-output initialization), SHA256 `d94791120cd73a5d63d9520ad16f875100e5bc9ed72fbab1c7a183316e1a6044`. It retains the 1024-wide residual MLP, frozen pretrained answer-code weights, seed 20261014, AdamW parameter groups/weight decay, learning rate, gradient clipping, warmup fraction and cosine schedule formula. The loss remains cross-entropy plus lambda times the per-example mean squared centered residual logits over valid options only.

The completed v4 27B cache contains 21,524 training views, with unchanged 809-question dev and 857-question calibration sets. With effective batch size 64, ten epochs give 3,370 optimizer updates per lambda (16,850 total), versus 1,840 per lambda in E9. This matches epoch count rather than optimizer update count; any comparison of the data versions also includes that difference in training duration. No backbone extraction is repeated.

For each lambda, select maximum native dev accuracy and the earliest epoch on ties. Select the winning lambda by maximum selected dev accuracy, then lower dev NLL, then smaller lambda. Lock this choice before accessing test features/labels. Fit temperature on calibration and evaluate the test panel only for the chosen checkpoint. No tests are evaluated for the other four candidates.

Each epoch records online training CE, residual penalty and combined objective, plus full training and dev accuracy, NLL, Brier score, calibration error and residual magnitude. The runtime checks valid-option masking, centering shift invariance and zero-point gradients. The post-run audit verifies all fifty epochs, exact initialization identity, objective decomposition, role counts and selection rules.

Durable metrics, run links and verification receipts are written to `outputs/training/v4-residual-l2-sweep`. The full checkpoint/optimizer archive is downloaded to `outputs/model-artifacts/v4-residual-l2-sweep.tar.gz` and SHA256-verified before any cleanup. See `results.md`, `postrun-audit.json` and `export.json` there for final status.
