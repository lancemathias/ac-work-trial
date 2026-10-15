# E5 / frozen pretrained equal probability mixture benchmark results

Locked temperature: **T = 0.83693784**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 72.04% | 72.04% | 0.6087 | 0.3292 | 0.0882 |
| Belebele | 200/200 | 94.00% | 94.00% | 0.1630 | 0.0795 | 0.0464 |
| Circa | 200/200 | 59.50% | 59.50% | 1.0929 | 0.5263 | 0.0753 |
| ContractNLI | 200/200 | 82.00% | 82.00% | 0.5377 | 0.2852 | 0.0873 |
| FinancialPhraseBank | 200/200 | 85.50% | 85.50% | 0.4069 | 0.2178 | 0.0779 |
| JevBench_public_hard | 111/111 | 72.07% | 72.07% | 0.6647 | 0.3674 | 0.0717 |
| JudgeBench | 200/200 | 80.00% | 80.00% | 0.4731 | 0.3036 | 0.0618 |
| RAGTruth | 200/200 | 62.50% | 62.50% | 0.6351 | 0.4457 | 0.0406 |
| TabFact | 200/200 | 68.50% | 68.50% | 0.5417 | 0.3687 | 0.0813 |
| TruthfulQA_binary | 200/200 | 72.00% | 72.00% | 0.5526 | 0.3708 | 0.0845 |
| WinoGrande | 200/200 | 79.50% | 79.50% | 0.4480 | 0.2879 | 0.0494 |
| **Macro** | | **75.24%** | **75.24%** | 0.5568 | 0.3256 | 0.0695 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
