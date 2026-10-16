# v4 residual L2 sweep; dev-selected winner benchmark results

Locked temperature: **T = 1.1755425**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 71.50% | 71.50% | 0.6779 | 0.3916 | 0.1103 |
| Belebele | 200/200 | 96.00% | 96.00% | 0.1389 | 0.0621 | 0.0497 |
| Circa | 200/200 | 76.50% | 76.50% | 0.8180 | 0.3538 | 0.1710 |
| ContractNLI | 200/200 | 82.00% | 82.00% | 0.5131 | 0.2788 | 0.0741 |
| FinancialPhraseBank | 200/200 | 80.00% | 80.00% | 0.4827 | 0.2643 | 0.0767 |
| JevBench_public_hard | 111/111 | 71.17% | 71.17% | 0.6954 | 0.3748 | 0.0719 |
| JudgeBench | 200/200 | 77.50% | 77.50% | 0.4855 | 0.3146 | 0.0640 |
| RAGTruth | 200/200 | 66.50% | 66.50% | 0.6235 | 0.4345 | 0.0293 |
| TabFact | 200/200 | 80.00% | 80.00% | 0.4053 | 0.2632 | 0.0445 |
| TruthfulQA_binary | 200/200 | 77.50% | 77.50% | 0.5412 | 0.3590 | 0.0939 |
| WinoGrande | 200/200 | 77.00% | 77.00% | 0.4431 | 0.2885 | 0.0555 |
| **Macro** | | **77.79%** | **77.79%** | 0.5295 | 0.3077 | 0.0765 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
