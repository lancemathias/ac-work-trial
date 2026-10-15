# Math source existing-source-control calibrated benchmark results

Locked temperature: **T = 1.3094417**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 56.46% | 56.46% | 0.9644 | 0.5348 | 0.0802 |
| Belebele | 200/200 | 81.00% | 81.00% | 0.5487 | 0.2771 | 0.0616 |
| Circa | 200/200 | 36.00% | 36.00% | 1.4410 | 0.7126 | 0.1696 |
| ContractNLI | 199/200 | 65.83% | 65.50% | 0.8335 | 0.5020 | 0.0581 |
| FinancialPhraseBank | 200/200 | 81.00% | 81.00% | 0.4616 | 0.2637 | 0.1030 |
| JevBench_public_hard | 111/111 | 47.75% | 47.75% | 1.1267 | 0.6558 | 0.1807 |
| JudgeBench | 200/200 | 63.00% | 63.00% | 0.6415 | 0.4514 | 0.0450 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.7200 | 0.5039 | 0.1692 |
| TabFact | 200/200 | 49.50% | 49.50% | 0.7271 | 0.5317 | 0.1134 |
| TruthfulQA_binary | 200/200 | 64.50% | 64.50% | 0.6890 | 0.4828 | 0.1078 |
| WinoGrande | 200/200 | 59.50% | 59.50% | 0.7614 | 0.5349 | 0.1904 |
| **Macro** | | **60.96%** | **60.93%** | 0.8105 | 0.4955 | 0.1163 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
