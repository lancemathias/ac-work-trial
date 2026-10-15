# Source add-back existing-source-control calibrated benchmark results

Locked temperature: **T = 1.4542226**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 52.78% | 52.78% | 1.0263 | 0.5679 | 0.1435 |
| Belebele | 200/200 | 80.00% | 80.00% | 0.5612 | 0.2845 | 0.0687 |
| Circa | 200/200 | 30.50% | 30.50% | 1.7270 | 0.8376 | 0.2463 |
| ContractNLI | 199/200 | 58.29% | 58.00% | 0.9216 | 0.5693 | 0.1706 |
| FinancialPhraseBank | 200/200 | 86.00% | 86.00% | 0.4182 | 0.2253 | 0.1271 |
| JevBench_public_hard | 111/111 | 42.34% | 42.34% | 1.2043 | 0.7011 | 0.2778 |
| JudgeBench | 200/200 | 64.00% | 64.00% | 0.6343 | 0.4435 | 0.0912 |
| RAGTruth | 200/200 | 63.50% | 63.50% | 0.6495 | 0.4589 | 0.0628 |
| TabFact | 200/200 | 47.50% | 47.50% | 0.9776 | 0.7182 | 0.3306 |
| TruthfulQA_binary | 200/200 | 58.50% | 58.50% | 0.6862 | 0.4887 | 0.0567 |
| WinoGrande | 200/200 | 61.50% | 61.50% | 0.7177 | 0.5036 | 0.1541 |
| **Macro** | | **58.63%** | **58.60%** | 0.8658 | 0.5271 | 0.1572 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
