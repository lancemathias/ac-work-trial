# Math source aegis2 calibrated benchmark results

Locked temperature: **T = 1.9614165**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 54.23% | 54.23% | 0.9577 | 0.5300 | 0.1354 |
| Belebele | 200/200 | 81.50% | 81.50% | 0.5921 | 0.2957 | 0.0943 |
| Circa | 200/200 | 37.50% | 37.50% | 1.4873 | 0.7199 | 0.1211 |
| ContractNLI | 199/200 | 68.34% | 68.00% | 0.8124 | 0.4810 | 0.1138 |
| FinancialPhraseBank | 200/200 | 89.00% | 89.00% | 0.4070 | 0.2162 | 0.1405 |
| JevBench_public_hard | 111/111 | 45.95% | 45.95% | 1.0205 | 0.6046 | 0.1092 |
| JudgeBench | 200/200 | 62.00% | 62.00% | 0.6436 | 0.4519 | 0.0416 |
| RAGTruth | 200/200 | 57.00% | 57.00% | 0.6650 | 0.4731 | 0.0534 |
| TabFact | 200/200 | 47.50% | 47.50% | 0.8106 | 0.6058 | 0.2292 |
| TruthfulQA_binary | 200/200 | 58.00% | 58.00% | 0.6817 | 0.4884 | 0.0344 |
| WinoGrande | 200/200 | 61.50% | 61.50% | 0.7119 | 0.4984 | 0.1305 |
| **Macro** | | **60.23%** | **60.20%** | 0.7991 | 0.4877 | 0.1094 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
