# Original random residual head: cached test comparison

W&B: https://wandb.ai/lmathias/ac-worktrial-decisions/runs/o2j7x479

The random condition uses the exact original saved Xavier-initialized 1024-wide MLP. No head weights are fitted. Each condition uses a positive scalar temperature fitted on the same nonbenchmark calibration partition; this does not change argmax accuracy. The same-prompt control zeroes the additive MLP correction and retains the identical cached backbone features and frozen code rows. The earlier readout uses its original prompt/runtime, so the same-prompt control is the closest architectural comparison.

| Benchmark | Random MLP | Fixed v3 | Permuted v3.1 | Readout: same prompt | Earlier readout | Earlier + reverse |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 70.47 | 69.93 | 71.98 | 77.05 | 75.42 | 75.42 |
| Belebele | 95.00 | 96.00 | 95.50 | 96.50 | 96.50 | 96.00 |
| Circa | 61.50 | 63.50 | 83.50 | 76.50 | 73.50 | 77.50 |
| ContractNLI | 79.00 | 69.50 | 70.50 | 81.00 | 79.00 | 80.50 |
| FinancialPhraseBank | 90.00 | 67.50 | 78.50 | 82.50 | 83.00 | 81.50 |
| JevBench_public_hard | 71.17 | 71.17 | 77.48 | 73.87 | 72.07 | 72.97 |
| JudgeBench | 73.00 | 77.50 | 76.00 | 77.50 | 76.00 | 77.50 |
| RAGTruth | 67.00 | 34.50 | 45.50 | 76.00 | 72.00 | 76.50 |
| TabFact | 73.00 | 78.50 | 83.00 | 78.00 | 73.00 | 83.00 |
| TruthfulQA_binary | 71.00 | 74.00 | 77.50 | 70.00 | 69.50 | 72.50 |
| WinoGrande | 75.00 | 72.50 | 75.50 | 79.50 | 78.50 | 81.00 |
| **Macro** | **75.10** | **70.42** | **75.91** | **78.95** | **77.14** | **79.49** |

No model/optimizer updates or new backbone inference. One initial head seed, 20261014. The trained v3.1 comparison additionally changes view weighting and total optimizer updates.
