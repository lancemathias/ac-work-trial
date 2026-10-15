# Frozen pretrained Qwen3.8-27B letter readout benchmark results

Locked temperature: **T = 1.3424165**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 75.42% | 75.42% | 0.5873 | 0.3241 | 0.0914 |
| Belebele | 200/200 | 96.50% | 96.50% | 0.1571 | 0.0682 | 0.0540 |
| Circa | 200/200 | 73.50% | 73.50% | 0.9936 | 0.4354 | 0.2378 |
| ContractNLI | 200/200 | 79.00% | 79.00% | 0.4782 | 0.2756 | 0.0755 |
| FinancialPhraseBank | 200/200 | 83.00% | 83.00% | 0.4645 | 0.2486 | 0.1106 |
| JevBench_public_hard | 111/111 | 72.07% | 72.07% | 0.7128 | 0.3926 | 0.0776 |
| JudgeBench | 200/200 | 76.00% | 76.00% | 0.4857 | 0.3151 | 0.0641 |
| RAGTruth | 200/200 | 72.00% | 72.00% | 0.5509 | 0.3649 | 0.1514 |
| TabFact | 200/200 | 73.00% | 73.00% | 0.5252 | 0.3523 | 0.0390 |
| TruthfulQA_binary | 200/200 | 69.50% | 69.50% | 0.5274 | 0.3560 | 0.0736 |
| WinoGrande | 200/200 | 78.50% | 78.50% | 0.4821 | 0.3139 | 0.0956 |
| **Macro** | | **77.14%** | **77.14%** | 0.5423 | 0.3133 | 0.0973 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
