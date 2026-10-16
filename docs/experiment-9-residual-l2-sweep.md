# Experiment 9: Centered-residual L2 sweep

User-directed follow-up to E8. User-specified coefficients: 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, plus a matched λ=0 control (nine fits total). Metrics and configurations are explicitly authorized for W&B publication; model checkpoints remain local/Modal artifacts. Each fresh ten-epoch fit loads the exact saved E8 zero-output initial state, uses the same cached v3.1 training views, batch order, AdamW and matrix weight decay, warmup/cosine schedule, gradient clipping, and native development partitions. No backbone inference or parameter updates occur.

For valid-option residual vector Δ, subtract its per-example mean. The penalty is the mean across examples of the mean squared centered entries, so each example has equal penalty weight irrespective of option count. Training minimizes CE(base + Δ, target) + λ × penalty. This penalizes corrections to predictions, separately from existing AdamW weight decay.

Log online CE, residual L2 and total objective separately, plus full checkpoint train/development NLL, accuracy, and correction RMS each epoch. Save all epoch weights and optimizer states. Within each coefficient, select maximum native development accuracy with earliest epoch ties, as in E8. Across coefficients, select maximum development accuracy, then lower development NLL, then smaller λ. Lock that choice in a committed-volume selection record before reading any test features or labels. Evaluate only that winning checkpoint on the exploratory existing panel. Hyperparameter selection on a repeatedly used development set remains a limitation.

The λ=0 run must reproduce E8's ten development accuracies and NLL within 1e-5; failure stops the sweep. Runtime contracts verify valid-option masking, invariance to common logit shifts, and finite gradients at zero. Every coefficient starts from the same checksummed state. Export the complete tensor/checkpoint/trace archive locally with SHA256 verification. Preserve all previous research artifacts.

Workspace `ac-worktrials`; profile `ac-worktrial`; environment `worktrial-lance`. Run: `.venv/bin/modal run --profile ac-worktrial --env worktrial-lance --detach scripts/modal_residual_l2_sweep.py`.

## Completed result

All nine coefficients completed ten epochs (16,560 total optimizer updates). The λ=0 control reproduced E8's development curve exactly. Development selection chose λ=0.3, epoch 8: accuracy 85.7849%, raw NLL 0.389121, centered correction RMS 0.211290 (versus 0.708309 for the unregularized selected checkpoint). The selected winner reached **76.7850%** benchmark composite, versus E8's **73.5239%** and matched frozen code logits **78.9479%**. Thus regularization recovered 3.26 points but remained 2.16 points below the base. Only the development-selected winner received new test evaluation. One seed on the repeatedly used development/test panels does not establish a general improvement.

[All W&B runs](https://wandb.ai/lmathias/ac-worktrial-decisions/groups/experiment-9-residual-l2-sweep); [selected λ=0.3 run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/8wzgubyg). W&B receives metrics/configurations only. Checkpoints and optimizer states are retained in Modal and exported locally with checksums.

The user designated this sweep **Experiment 9**, with E8 as its parent. W&B runs and group use Experiment 9. The launch, Modal directory, archives, and source-bound metric paths retain `experiment-8-residual-l2-sweep-v2` as their immutable legacy ID. Canonical descriptor: `configs/experiment-9-residual-l2-sweep.json`.
