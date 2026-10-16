# v4 residual L2 sweep

E9 protocol; 10 epochs per lambda, same saved zero-output initialization and optimizer settings. v4 has 21,524 training views; unchanged dev/calibration and benchmark panel. Metrics/configuration published to W&B; checkpoints preserved locally.

| Lambda | Best epoch | Dev accuracy | Train accuracy | Dev NLL | W&B |
|---:|---:|---:|---:|---:|---|
| 0.03 | 5 | 85.66% | 82.94% | 0.3835 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/5ymkezua) |
| 0.1 | 2 | 85.54% | 80.20% | 0.3987 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/lc4ktopr) |
| 0.3 | 8 | 85.54% | 82.94% | 0.3873 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/ktxwmsgd) |
| 1 | 7 | 85.66% | 79.08% | 0.3938 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/omcul47u) |
| 3 | 7 | 84.80% | 76.68% | 0.4014 | [run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/nhbpik7g) |

Selected lambda: **0.03**, epoch **5**, using dev accuracy, then dev NLL, then smaller lambda. Only the selected checkpoint was tested.

Benchmark macro accuracy: **77.79%**.

All 50 epochs log online training CE, residual penalty/objective, full training evaluation and dev metrics. The 10-epoch schedule yields 3,370 optimizer updates per lambda, versus 1,840 in E9 because v4 is larger.

E9 selected winner: 76.78% test macro accuracy; v4 difference +1.00 percentage points. RAGTruth: 66.5% versus 52.5% in E9. This comparison does not isolate the effect of added data.
