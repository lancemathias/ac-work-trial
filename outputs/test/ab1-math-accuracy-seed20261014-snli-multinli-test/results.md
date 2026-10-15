# Math source snli-multinli calibrated benchmark results

Locked temperature: **T = 1.4562733**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 54.17% | 54.17% | 0.9735 | 0.5419 | 0.1149 |
| Belebele | 200/200 | 81.50% | 81.50% | 0.5490 | 0.2765 | 0.0712 |
| Circa | 200/200 | 35.50% | 35.50% | 1.4280 | 0.7196 | 0.1785 |
| ContractNLI | 199/200 | 65.83% | 65.50% | 0.7522 | 0.4461 | 0.1066 |
| FinancialPhraseBank | 200/200 | 85.50% | 85.50% | 0.4924 | 0.2735 | 0.1454 |
| JevBench_public_hard | 111/111 | 46.85% | 46.85% | 1.1485 | 0.6627 | 0.1892 |
| JudgeBench | 200/200 | 64.50% | 64.50% | 0.6205 | 0.4306 | 0.0547 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.6651 | 0.4691 | 0.1300 |
| TabFact | 200/200 | 47.50% | 47.50% | 0.7808 | 0.5791 | 0.2073 |
| TruthfulQA_binary | 200/200 | 64.00% | 64.00% | 0.6746 | 0.4748 | 0.0808 |
| WinoGrande | 200/200 | 60.50% | 60.50% | 0.7285 | 0.5113 | 0.1451 |
| **Macro** | | **61.08%** | **61.05%** | 0.8012 | 0.4896 | 0.1294 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
