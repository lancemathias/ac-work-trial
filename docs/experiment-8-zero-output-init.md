# Experiment 8: Zero-initialized residual output

User-directed October 15, 2026. Retrain the E6 architecture on the exact v3.1 cached feature views. Load E6's original saved initial state, then zero only `down.weight` and `down.bias`; preserve LayerNorm and the random hidden projection. This is a fresh fit, not a continuation of E6's trained checkpoint.

Hold E6's data, optimizer, seed, batch size, learning-rate schedule, five epochs (920 updates), maximum native dev accuracy selection (earliest ties), and calibration-only temperature fitting fixed. The original E6 test composite is 75.91%. Results remain exploratory on the repeatedly observed panel.

The runner verifies exact base-logit parity at initialization and checks that the output projection receives a gradient on the first update and the hidden projection on the second. It saves initial dev metrics, all epoch metrics and optimizer checkpoints, selected weights, calibration, test predictions, and frozen base logits on identical test prompts. No backbone extraction is needed.

Run with `.venv/bin/modal run --profile ac-worktrial --env worktrial-lance --detach scripts/modal_e6_zero_init.py`. All resources use workspace `ac-worktrials`, profile `ac-worktrial`, environment `worktrial-lance`. The runner exports a checksummed archive to `outputs/model-artifacts/` and small receipts to `outputs/autoresearch/`. Existing experiments and artifacts are preserved.

Experiment number: **8**. E6 identifies the comparator, not this experiment. W&B run: [Experiment 8](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/qzctuapq). The already launched storage ID `e6-zero-output-init-v31-seed20261014` is retained for provenance; canonical ID: `experiment-8-zero-output-init-v31-seed20261014`.
