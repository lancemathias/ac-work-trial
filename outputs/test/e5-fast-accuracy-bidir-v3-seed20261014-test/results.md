# e5-fast-accuracy-bidir-v3-seed20261014 bidirectional_full_attention calibrated benchmark results

Locked temperature: **T = 1.0645429**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 69.69% | 69.69% | 0.7728 | 0.4235 | 0.0977 |
| Belebele | 200/200 | 93.50% | 93.50% | 0.2614 | 0.1227 | 0.0519 |
| Circa | 200/200 | 50.00% | 50.00% | 1.6455 | 0.7287 | 0.2393 |
| ContractNLI | 199/200 | 64.82% | 64.50% | 0.8912 | 0.4782 | 0.1351 |
| FinancialPhraseBank | 200/200 | 86.00% | 86.00% | 0.4521 | 0.2389 | 0.1014 |
| JevBench_public_hard | 111/111 | 69.37% | 69.37% | 0.7107 | 0.4049 | 0.0922 |
| JudgeBench | 200/200 | 78.00% | 78.00% | 0.5240 | 0.3435 | 0.0756 |
| RAGTruth | 200/200 | 41.00% | 41.00% | 0.8727 | 0.6508 | 0.2997 |
| TabFact | 200/200 | 63.50% | 63.50% | 0.6024 | 0.4184 | 0.0556 |
| TruthfulQA_binary | 200/200 | 59.00% | 59.00% | 0.6657 | 0.4719 | 0.0802 |
| WinoGrande | 200/200 | 78.00% | 78.00% | 0.4845 | 0.3110 | 0.0638 |
| **Macro** | | **68.44%** | **68.41%** | 0.7166 | 0.4175 | 0.1175 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
