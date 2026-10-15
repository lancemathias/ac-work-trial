# Source add-back existing-source-control calibrated benchmark results

Locked temperature: **T = 1.8041798**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 53.20% | 53.20% | 0.9723 | 0.5459 | 0.1276 |
| Belebele | 200/200 | 81.00% | 81.00% | 0.5610 | 0.2823 | 0.0597 |
| Circa | 200/200 | 40.50% | 40.50% | 1.4462 | 0.6891 | 0.1428 |
| ContractNLI | 199/200 | 58.79% | 58.50% | 0.8694 | 0.5237 | 0.1423 |
| FinancialPhraseBank | 200/200 | 91.50% | 91.50% | 0.4034 | 0.2098 | 0.1630 |
| JevBench_public_hard | 111/111 | 44.14% | 44.14% | 1.1250 | 0.6641 | 0.2041 |
| JudgeBench | 200/200 | 65.50% | 65.50% | 0.6315 | 0.4406 | 0.0674 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.6462 | 0.4546 | 0.1045 |
| TabFact | 200/200 | 47.50% | 47.50% | 0.8367 | 0.6226 | 0.2448 |
| TruthfulQA_binary | 200/200 | 61.00% | 61.00% | 0.6795 | 0.4804 | 0.0578 |
| WinoGrande | 200/200 | 60.50% | 60.50% | 0.7062 | 0.4977 | 0.1284 |
| **Macro** | | **60.88%** | **60.85%** | 0.8070 | 0.4919 | 0.1311 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
