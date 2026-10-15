# E1 calibrated benchmark results

Locked temperature: **T = 0.79510347**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 55.86% | 55.86% | 1.0954 | 0.5760 | 0.1583 |
| Belebele | 200/200 | 81.00% | 81.00% | 0.5865 | 0.2956 | 0.0987 |
| Circa | 200/200 | 61.00% | 61.00% | 1.2510 | 0.5907 | 0.1912 |
| ContractNLI | 199/200 | 53.27% | 53.00% | 0.8565 | 0.5099 | 0.1302 |
| FinancialPhraseBank | 200/200 | 48.00% | 48.00% | 0.8666 | 0.5775 | 0.1822 |
| JevBench_public_hard | 111/111 | 45.05% | 45.05% | 1.1066 | 0.6574 | 0.1917 |
| JudgeBench | 200/200 | 44.50% | 44.50% | 0.8976 | 0.6659 | 0.3394 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.7261 | 0.5057 | 0.1608 |
| TabFact | 200/200 | 56.50% | 56.50% | 0.6918 | 0.4979 | 0.0433 |
| TruthfulQA_binary | 200/200 | 60.00% | 60.00% | 0.6785 | 0.4826 | 0.0770 |
| WinoGrande | 200/200 | 60.00% | 60.00% | 0.7447 | 0.5185 | 0.1430 |
| **Macro** | | **57.38%** | **57.35%** | 0.8638 | 0.5343 | 0.1560 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
