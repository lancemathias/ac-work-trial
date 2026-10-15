# E4 accuracy-selected Qwen3.8-27B benchmark results

Locked temperature: **T = 2.0035164**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 58.70% | 58.70% | 0.8738 | 0.4825 | 0.1502 |
| Belebele | 200/200 | 89.00% | 89.00% | 0.3282 | 0.1652 | 0.0672 |
| Circa | 200/200 | 62.00% | 62.00% | 1.1275 | 0.5047 | 0.0788 |
| ContractNLI | 199/200 | 52.26% | 52.00% | 0.9770 | 0.5628 | 0.2222 |
| FinancialPhraseBank | 200/200 | 47.00% | 47.00% | 1.0013 | 0.6112 | 0.2080 |
| JevBench_public_hard | 111/111 | 58.56% | 58.56% | 0.8910 | 0.5092 | 0.1231 |
| JudgeBench | 200/200 | 70.50% | 70.50% | 0.5570 | 0.3751 | 0.0589 |
| RAGTruth | 200/200 | 62.50% | 62.50% | 0.6399 | 0.4487 | 0.0669 |
| TabFact | 200/200 | 51.50% | 51.50% | 0.6791 | 0.4869 | 0.0890 |
| TruthfulQA_binary | 200/200 | 45.50% | 45.50% | 0.7294 | 0.5353 | 0.1626 |
| WinoGrande | 200/200 | 73.00% | 73.00% | 0.5574 | 0.3631 | 0.0802 |
| **Macro** | | **60.96%** | **60.93%** | 0.7601 | 0.4586 | 0.1188 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
