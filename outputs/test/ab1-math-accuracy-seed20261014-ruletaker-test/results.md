# Math source ruletaker calibrated benchmark results

Locked temperature: **T = 1.8375222**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 55.13% | 55.13% | 0.9525 | 0.5318 | 0.0866 |
| Belebele | 200/200 | 79.00% | 79.00% | 0.5789 | 0.2909 | 0.0626 |
| Circa | 200/200 | 37.50% | 37.50% | 1.4974 | 0.7020 | 0.1351 |
| ContractNLI | 199/200 | 66.33% | 66.00% | 0.9039 | 0.5202 | 0.1296 |
| FinancialPhraseBank | 200/200 | 88.50% | 88.50% | 0.4551 | 0.2407 | 0.1798 |
| JevBench_public_hard | 111/111 | 44.14% | 44.14% | 1.1235 | 0.6556 | 0.2430 |
| JudgeBench | 200/200 | 65.00% | 65.00% | 0.6375 | 0.4461 | 0.1080 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.6548 | 0.4614 | 0.0946 |
| TabFact | 200/200 | 47.00% | 47.00% | 0.7689 | 0.5694 | 0.1920 |
| TruthfulQA_binary | 200/200 | 64.00% | 64.00% | 0.6666 | 0.4714 | 0.0685 |
| WinoGrande | 200/200 | 58.00% | 58.00% | 0.6843 | 0.4858 | 0.1229 |
| **Macro** | | **60.96%** | **60.93%** | 0.8112 | 0.4887 | 0.1293 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
