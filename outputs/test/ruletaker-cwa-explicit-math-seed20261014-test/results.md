# Same-record explicit closed-world RuleTaker benchmark results

Locked temperature: **T = 1.4505957**, fitted on the separate training-source calibration split.

Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.

| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200/200 | 55.74% | 55.74% | 0.9899 | 0.5508 | 0.1135 |
| Belebele | 200/200 | 81.50% | 81.50% | 0.5594 | 0.2819 | 0.0558 |
| Circa | 200/200 | 36.50% | 36.50% | 1.5035 | 0.7268 | 0.1775 |
| ContractNLI | 199/200 | 63.82% | 63.50% | 0.8377 | 0.5078 | 0.0796 |
| FinancialPhraseBank | 200/200 | 90.00% | 90.00% | 0.4349 | 0.2269 | 0.1727 |
| JevBench_public_hard | 111/111 | 45.95% | 45.95% | 1.1737 | 0.6804 | 0.2081 |
| JudgeBench | 200/200 | 63.00% | 63.00% | 0.6394 | 0.4490 | 0.0742 |
| RAGTruth | 200/200 | 66.00% | 66.00% | 0.6681 | 0.4723 | 0.1374 |
| TabFact | 200/200 | 47.50% | 47.50% | 0.9000 | 0.6689 | 0.2904 |
| TruthfulQA_binary | 200/200 | 55.50% | 55.50% | 0.6870 | 0.4884 | 0.0992 |
| WinoGrande | 200/200 | 60.50% | 60.50% | 0.7056 | 0.4976 | 0.1278 |
| **Macro** | | **60.55%** | **60.52%** | 0.8272 | 0.5046 | 0.1397 |

Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.
