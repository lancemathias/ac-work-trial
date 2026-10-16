# Frozen pretrained native/reverse probability pool benchmark results

Locked temperature: **T = 0.98872331**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 75.42% | 75.42% | 0.6527 | 0.3542 | 0.1111 |
| Belebele | 200/200 | 96.00% | 96.00% | 0.1262 | 0.0600 | 0.0369 |
| Circa | 200/200 | 77.50% | 77.50% | 0.7523 | 0.3194 | 0.1644 |
| ContractNLI | 200/200 | 80.50% | 80.50% | 0.4614 | 0.2602 | 0.0545 |
| FinancialPhraseBank | 200/200 | 81.50% | 81.50% | 0.4022 | 0.2232 | 0.0852 |
| JevBench_public_hard | 111/111 | 72.97% | 72.97% | 0.6526 | 0.3560 | 0.0756 |
| JudgeBench | 200/200 | 77.50% | 77.50% | 0.5595 | 0.3746 | 0.1247 |
| RAGTruth | 200/200 | 76.50% | 76.50% | 0.5419 | 0.3576 | 0.1165 |
| TabFact | 200/200 | 83.00% | 83.00% | 0.4222 | 0.2657 | 0.0913 |
| TruthfulQA_binary | 200/200 | 72.50% | 72.50% | 0.5248 | 0.3532 | 0.0374 |
| WinoGrande | 200/200 | 81.00% | 81.00% | 0.4556 | 0.2910 | 0.0931 |
| **Macro** | | **79.49%** | **79.49%** | 0.5047 | 0.2923 | 0.0901 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
