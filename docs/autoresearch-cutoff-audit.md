# Autoresearch cutoff audit

Policy: experiments started after **October 15, 2026, 10:30 a.m. Pacific (17:30 UTC)** must not carry an `autoresearch` tag. v3.1 is user-directed. Retrospective imports inherit the classification of the original experiment; their import timestamp is not an experiment launch.

Inspected **66 W&B runs**. No post-cutoff non-import run currently carries an autoresearch tag. The five non-import records created after the cutoff include the handoff audit itself. The initial audit was read-only. The fallback E4 retrospective record is user-directed; its optimized-kernel follow-up is autoresearch per the user’s clarification. Metrics are unchanged.

## Post-cutoff non-import records

| Run | Created (Pacific) | Tags |
| --- | --- | --- |
| [residual-readout1024-qwen38-v3-order-seed20261014](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/c09mctl9) | 12:03:41 | None |
| [Research handoff — accepted checkpoints and evidence](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/handoff20261015) | 13:07:49 | None |
| [readout1024-qwen38-v3.1-choice-permutation1-seed20261014](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/28lh244a) | 14:01:39 | None |
| [random-residual1024-qwen38-test-seed20261014](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/o2j7x479) | 14:15:16 | None |
| [pplx-recipe-qwen35-9b-10k-2epoch-seed20260920](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/n9t3qp79) | 14:41:42 | None |

## Flagged for provenance review: later retrospective imports

These **16 records** have autoresearch tags and were created after the cutoff, but each is explicitly a historical-evaluation import (`inference_performed_by_import=false`). They are flags for timestamp/provenance review, not confirmed newly started autoresearch experiments. Their configs identify the original source run and saved metric hash.

| Imported condition | W&B record | Original source run |
| --- | --- | --- |
| Evaluation / Bidirectional full attention | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist4d6cf18546e78e893ffd) | `e5-fast-accuracy-bidir-v3-seed20261014-test` |
| Evaluation / Native 27B | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist3b6936c086ab7724e134) | `readout-fast-qwen38-27b-v1` |
| Evaluation / Native + reverse | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist35a890f38ba716b97208) | `readout-native-reverse-pool-v1` |
| Evaluation / Source control | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist468deec18224c04c1a7d) | `ab1-math-accuracy-seed20261014-control-1-test` |
| Evaluation / RuleTaker | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/histdbc46cf0d91aa5c199cc) | `ab1-math-accuracy-seed20261014-ruletaker-test` |
| Evaluation / Aegis2 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist3b62f16b84e65daea91a) | `ab1-math-accuracy-seed20261014-aegis2-test` |
| Evaluation / CivilComments | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist192d62ebd6098fe15cad) | `ab1-math-accuracy-seed20261014-civilcomments-test` |
| Evaluation / SNLI + MultiNLI | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist7e7ac1ac75c3d5f1e5e3) | `ab1-math-accuracy-seed20261014-snli-multinli-test` |
| Evaluation / MASSIVE | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist2d89320e8fb746cfc035) | `ab1-math-accuracy-seed20261014-massive-test` |
| Evaluation / BANKING77 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist469cc7b5d85fa89d47b4) | `ab1-math-accuracy-seed20261014-banking77-test` |
| Evaluation / RuleTaker repaired | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist6fe20aaa43b64beefa3e) | `ruletaker-cwa-explicit-math-seed20261014-test` |
| Evaluation / Joint-head + native mixture | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/histbb478856e84c09dc39d4) | `cached-e5-readout-equal-mixture-v1` |
| Evaluation / Auto head control repeat | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hista73e1f384c02c3e2964a) | `ab1-fast-accuracy-seed20261014-existing-source-control-repeat-test` |
| Evaluation / Auto head control 1 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/histddb53a0a91d41b62eeb5) | `ab1-fast-accuracy-seed20261014-existing-source-control-test` |
| Evaluation / Math head control repeat | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist39e2f0458ca6bf3ab972) | `ab1-math-accuracy-seed20261014-control-2-test` |
| Evaluation / E4 optimized | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist7df299ca09e4563b12d9) | `e4-fast-accuracy-v3-seed20261014-test` |
