# Math source massive calibrated benchmark results

Locked temperature: **T = 1.7216459**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 56.58% | 56.58% | 0.9896 | 0.5465 | 0.1351 |
| Belebele | 200/200 | 82.00% | 82.00% | 0.5615 | 0.2842 | 0.0558 |
| Circa | 200/200 | 42.50% | 42.50% | 1.3402 | 0.6508 | 0.1490 |
| ContractNLI | 199/200 | 65.83% | 65.50% | 0.8007 | 0.4726 | 0.0805 |
| FinancialPhraseBank | 200/200 | 88.50% | 88.50% | 0.4836 | 0.2562 | 0.1998 |
| JevBench_public_hard | 111/111 | 42.34% | 42.34% | 1.1661 | 0.6828 | 0.2377 |
| JudgeBench | 200/200 | 67.50% | 67.50% | 0.6184 | 0.4286 | 0.1225 |
| RAGTruth | 200/200 | 65.50% | 65.50% | 0.6549 | 0.4635 | 0.0996 |
| TabFact | 200/200 | 47.50% | 47.50% | 0.8851 | 0.6592 | 0.2858 |
| TruthfulQA_binary | 200/200 | 64.00% | 64.00% | 0.6876 | 0.4819 | 0.0712 |
| WinoGrande | 200/200 | 60.50% | 60.50% | 0.7254 | 0.5089 | 0.1669 |
| **Macro** | | **62.07%** | **62.04%** | 0.8103 | 0.4941 | 0.1458 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
