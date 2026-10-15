# Experiment 8 benchmark results

Locked temperature: **T = 1.2475466**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 70.59% | 70.59% | 0.6110 | 0.3416 | 0.0664 |
| Belebele | 200/200 | 96.00% | 96.00% | 0.1383 | 0.0625 | 0.0362 |
| Circa | 200/200 | 76.00% | 76.00% | 0.8946 | 0.3893 | 0.2208 |
| ContractNLI | 200/200 | 69.00% | 69.00% | 0.7484 | 0.4175 | 0.0922 |
| FinancialPhraseBank | 200/200 | 80.00% | 80.00% | 0.4981 | 0.2729 | 0.0828 |
| JevBench_public_hard | 111/111 | 71.17% | 71.17% | 0.6864 | 0.3691 | 0.1113 |
| JudgeBench | 200/200 | 75.00% | 75.00% | 0.5392 | 0.3426 | 0.0664 |
| RAGTruth | 200/200 | 38.50% | 38.50% | 0.8852 | 0.6664 | 0.3369 |
| TabFact | 200/200 | 82.50% | 82.50% | 0.3787 | 0.2431 | 0.0558 |
| TruthfulQA_binary | 200/200 | 74.50% | 74.50% | 0.5766 | 0.3782 | 0.0826 |
| WinoGrande | 200/200 | 75.50% | 75.50% | 0.4665 | 0.3078 | 0.0784 |
| **Macro** | | **73.52%** | **73.52%** | 0.5839 | 0.3446 | 0.1118 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
