# v3.1 cached-choice augmentation results

[W&B run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/28lh244a)

| Benchmark | Fixed v3 | v3.1 | Delta pp |
|---|---:|---:|---:|
| BBH | 69.93% | 71.98% | +2.05 |
| Belebele | 96.00% | 95.50% | -0.50 |
| Circa | 63.50% | 83.50% | +20.00 |
| ContractNLI | 69.50% | 70.50% | +1.00 |
| FinancialPhraseBank | 67.50% | 78.50% | +11.00 |
| JevBench_public_hard | 71.17% | 77.48% | +6.31 |
| JudgeBench | 77.50% | 76.00% | -1.50 |
| RAGTruth | 34.50% | 45.50% | +11.00 |
| TabFact | 78.50% | 83.00% | +4.50 |
| TruthfulQA_binary | 74.00% | 77.50% | +3.50 |
| WinoGrande | 72.50% | 75.50% | +3.00 |
| **Macro** | **70.42%** | **75.91%** | **+5.49** |

Selected epoch: 5. Training views: 11,755. Five epochs, 920 updates; fixed v3 had 490 updates. Same saved initial head and optimizer hyperparameters; native dev accuracy checkpoint selection and independent calibration temperature. This single-seed contrast changes augmentation, choice weighting, and total updates. All benchmark features were reused; no backbone inference.
