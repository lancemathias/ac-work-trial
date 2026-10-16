# Experiment 8: Zero-initialized residual output

User-directed October 15, 2026. Retrain the E6 architecture on the exact v3.1 cached feature views. Load E6's original saved initial state, then zero only `down.weight` and `down.bias`; preserve LayerNorm and the random hidden projection. This is a fresh fit, not a continuation of E6's trained checkpoint.

Hold E6's data, optimizer, seed, batch size, learning-rate schedule, five epochs (920 updates), maximum native dev accuracy selection (earliest ties), and calibration-only temperature fitting fixed. The original E6 test composite is 75.91%. Results remain exploratory on the repeatedly observed panel.

The runner verifies exact base-logit parity at initialization and checks that the output projection receives a gradient on the first update and the hidden projection on the second. It saves initial dev metrics, all epoch metrics and optimizer checkpoints, selected weights, calibration, test predictions, and frozen base logits on identical test prompts. No backbone extraction is needed.

Run with `.venv/bin/modal run --profile ac-worktrial --env worktrial-lance --detach scripts/modal_e6_zero_init.py`. All resources use workspace `ac-worktrials`, profile `ac-worktrial`, environment `worktrial-lance`. The runner exports a checksummed archive to `outputs/model-artifacts/` and small receipts to `outputs/autoresearch/`. Existing experiments and artifacts are preserved.

Experiment number: **8**. E6 identifies the comparator, not this experiment. W&B run: [Experiment 8](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/qzctuapq). The already launched storage ID `e6-zero-output-init-v31-seed20261014` is retained for provenance; canonical ID: `experiment-8-zero-output-init-v31-seed20261014`.

## Ten-epoch rerun

The user requested a fresh ten-epoch run, preserving the five-epoch attempt. Run ID: `experiment-8-zero-output-init-v31-10epochs-seed20261014`. The schedule spans 1,840 updates. Every epoch stores online training NLL, full-training checkpoint NLL/accuracy, and development NLL/accuracy. W&B history is recorded offline and exported locally because external model/result publication was rejected by automatic approval review.

The ten-epoch run completed all 1,840 updates. Maximum development accuracy selects epoch 8 (85.1669%; epoch 9 ties). Test composite: **73.5239%**, versus **78.9479%** for frozen logits on identical code prompts and **79.5671%** for the five-epoch attempt (selected epoch 1). Original E6 scored **75.9073%**. The five-versus-ten contrast changes both total updates and the cosine/warmup schedule; it is a fresh fit rather than continuation at a fixed schedule.

Full-training checkpoint NLL falls from 0.419593 to 0.343020 across ten epochs. Development NLL starts at 0.400643, reaches its minimum at epoch 4 (0.386831), and ends at 0.389185. Epoch 8 has train NLL 0.346014 and dev NLL 0.391163. The training/development mixture differs (training includes duplicated Choice permutation views), so the absolute gap alone is not a clean overfitting estimate. Falling training loss, largely flat development loss, and worse benchmark transfer together support harmful specialization during training. Zero initialization protects the initial outputs, not subsequent updates.

The user subsequently explicitly authorized syncing ten-epoch metrics and configuration to W&B, excluding checkpoints. `scripts/modal_experiment8_sync.py` publishes the ten saved epoch records, train-loss alias, selected epoch, and benchmark summaries. The local tensor archive remains the durable checkpoint export.
