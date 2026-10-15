# E3 Medium v3 calibrated benchmark results

Locked temperature: **T = 1.1069017**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 56.04% | 56.04% | 0.9688 | 0.5342 | 0.1218 |
| Belebele | 200/200 | 80.50% | 80.50% | 0.5385 | 0.2711 | 0.0655 |
| Circa | 200/200 | 31.50% | 31.50% | 1.7483 | 0.8377 | 0.2544 |
| ContractNLI | 199/200 | 66.33% | 66.00% | 0.8480 | 0.5050 | 0.0725 |
| FinancialPhraseBank | 200/200 | 80.50% | 80.50% | 0.4762 | 0.2731 | 0.0944 |
| JevBench_public_hard | 111/111 | 42.34% | 42.34% | 1.1598 | 0.6668 | 0.2184 |
| JudgeBench | 200/200 | 65.00% | 65.00% | 0.6382 | 0.4471 | 0.0497 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.7337 | 0.5124 | 0.1911 |
| TabFact | 200/200 | 47.00% | 47.00% | 0.7281 | 0.5331 | 0.1380 |
| TruthfulQA_binary | 200/200 | 63.50% | 63.50% | 0.6833 | 0.4804 | 0.0787 |
| WinoGrande | 200/200 | 60.00% | 60.00% | 0.8012 | 0.5518 | 0.2051 |
| **Macro** | | **59.88%** | **59.85%** | 0.8477 | 0.5102 | 0.1354 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
