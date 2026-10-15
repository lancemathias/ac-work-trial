# e4-fast-accuracy-v3-seed20261014 native_causal calibrated benchmark results

Locked temperature: **T = 2.0305022**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 61.11% | 61.11% | 0.8600 | 0.4773 | 0.1271 |
| Belebele | 200/200 | 88.50% | 88.50% | 0.3441 | 0.1732 | 0.0600 |
| Circa | 200/200 | 60.50% | 60.50% | 1.1457 | 0.5164 | 0.0951 |
| ContractNLI | 199/200 | 51.76% | 51.50% | 1.0367 | 0.5956 | 0.2405 |
| FinancialPhraseBank | 200/200 | 43.50% | 43.50% | 1.1111 | 0.6919 | 0.2686 |
| JevBench_public_hard | 111/111 | 61.26% | 61.26% | 0.8974 | 0.5022 | 0.1065 |
| JudgeBench | 200/200 | 69.50% | 69.50% | 0.5648 | 0.3820 | 0.0561 |
| RAGTruth | 200/200 | 61.50% | 61.50% | 0.6494 | 0.4578 | 0.0259 |
| TabFact | 200/200 | 55.50% | 55.50% | 0.6803 | 0.4883 | 0.1114 |
| TruthfulQA_binary | 200/200 | 46.50% | 46.50% | 0.7176 | 0.5241 | 0.1334 |
| WinoGrande | 200/200 | 74.50% | 74.50% | 0.5433 | 0.3519 | 0.0721 |
| **Macro** | | **61.28%** | **61.26%** | 0.7773 | 0.4692 | 0.1179 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
