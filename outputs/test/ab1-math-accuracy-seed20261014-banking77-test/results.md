# Math source banking77 calibrated benchmark results

Locked temperature: **T = 1.6944879**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 55.25% | 55.25% | 1.0008 | 0.5603 | 0.1417 |
| Belebele | 200/200 | 81.50% | 81.50% | 0.5724 | 0.2915 | 0.0593 |
| Circa | 200/200 | 31.50% | 31.50% | 1.6510 | 0.8109 | 0.2582 |
| ContractNLI | 199/200 | 67.84% | 67.50% | 0.8857 | 0.5018 | 0.1073 |
| FinancialPhraseBank | 200/200 | 88.50% | 88.50% | 0.4682 | 0.2403 | 0.1858 |
| JevBench_public_hard | 111/111 | 41.44% | 41.44% | 1.2019 | 0.6965 | 0.2583 |
| JudgeBench | 200/200 | 65.50% | 65.50% | 0.6323 | 0.4418 | 0.0588 |
| RAGTruth | 200/200 | 65.50% | 65.50% | 0.6447 | 0.4539 | 0.0748 |
| TabFact | 200/200 | 47.00% | 47.00% | 0.8796 | 0.6538 | 0.2786 |
| TruthfulQA_binary | 200/200 | 60.50% | 60.50% | 0.6930 | 0.4890 | 0.0618 |
| WinoGrande | 200/200 | 60.00% | 60.00% | 0.7304 | 0.5114 | 0.1444 |
| **Macro** | | **60.41%** | **60.38%** | 0.8509 | 0.5137 | 0.1481 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
