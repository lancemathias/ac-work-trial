# E2 Qwen3.5-9B v2 calibrated benchmark results

Locked temperature: **T = 1.5191406**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 60.87% | 60.87% | 0.9863 | 0.5457 | 0.0755 |
| Belebele | 200/200 | 79.00% | 79.00% | 0.5943 | 0.2975 | 0.0989 |
| Circa | 200/200 | 34.50% | 34.50% | 1.4123 | 0.6859 | 0.0591 |
| ContractNLI | 199/200 | 59.80% | 59.50% | 0.8847 | 0.5299 | 0.0656 |
| FinancialPhraseBank | 200/200 | 78.50% | 78.50% | 0.5917 | 0.3320 | 0.2049 |
| JevBench_public_hard | 111/111 | 47.75% | 47.75% | 1.0550 | 0.6254 | 0.0972 |
| JudgeBench | 200/200 | 63.50% | 63.50% | 0.6601 | 0.4680 | 0.0943 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.6874 | 0.4858 | 0.1532 |
| TabFact | 200/200 | 47.50% | 47.50% | 0.7790 | 0.5769 | 0.2038 |
| TruthfulQA_binary | 200/200 | 62.50% | 62.50% | 0.6707 | 0.4741 | 0.0552 |
| WinoGrande | 200/200 | 62.50% | 62.50% | 0.6695 | 0.4694 | 0.0604 |
| **Macro** | | **60.22%** | **60.19%** | 0.8174 | 0.4991 | 0.1062 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
