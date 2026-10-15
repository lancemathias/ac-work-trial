# Math source civilcomments calibrated benchmark results

Locked temperature: **T = 1.8704623**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 55.13% | 55.13% | 0.9856 | 0.5395 | 0.1043 |
| Belebele | 200/200 | 80.50% | 80.50% | 0.5575 | 0.2813 | 0.0580 |
| Circa | 200/200 | 38.50% | 38.50% | 1.5100 | 0.7139 | 0.1800 |
| ContractNLI | 199/200 | 66.83% | 66.50% | 0.8823 | 0.4894 | 0.0799 |
| FinancialPhraseBank | 200/200 | 87.50% | 87.50% | 0.4039 | 0.2218 | 0.0969 |
| JevBench_public_hard | 111/111 | 42.34% | 42.34% | 1.2148 | 0.6917 | 0.2648 |
| JudgeBench | 200/200 | 63.50% | 63.50% | 0.6292 | 0.4383 | 0.0501 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.7311 | 0.5057 | 0.1841 |
| TabFact | 200/200 | 52.00% | 52.00% | 0.7001 | 0.5063 | 0.0546 |
| TruthfulQA_binary | 200/200 | 63.50% | 63.50% | 0.7355 | 0.5108 | 0.1361 |
| WinoGrande | 200/200 | 60.50% | 60.50% | 0.7486 | 0.5131 | 0.1662 |
| **Macro** | | **61.48%** | **61.45%** | 0.8271 | 0.4920 | 0.1250 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
