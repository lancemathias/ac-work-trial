# Baselines — measured results

| Benchmark | n | PPLX | CLEF-flash | CLEF | Jev |
|---|---:|---:|---:|---:|---:|
| BBH | 200 | Pending | Pending | Pending | 92.51% (200/200) |
| Belebele | 200 | Pending | Pending | Pending | 97.50% (200/200) |
| Circa | 200 | Pending | Pending | Pending | 79.00% (200/200) |
| ContractNLI | 200 | Pending | Pending | Pending | 78.00% (200/200) |
| FinancialPhraseBank | 200 | Pending | Pending | Pending | 96.00% (200/200) |
| JevBench_public_hard | 111 | Pending | Pending | Pending | 72.97% (111/111) |
| JudgeBench | 200 | Pending | Pending | Pending | 78.50% (200/200) |
| RAGTruth | 200 | Pending | Pending | Pending | 83.50% (200/200) |
| TabFact | 200 | Pending | Pending | Pending | 86.50% (200/200) |
| TruthfulQA_binary | 200 | Pending | Pending | Pending | 88.50% (200/200) |
| WinoGrande | 200 | Pending | Pending | Pending | 91.00% (200/200) |
| **11-suite macro** | **2111** | Incomplete | Incomplete | Incomplete | 85.82% |

Cells report accuracy and valid/expected coverage. BBH uses its subtask macro-average. Pending is not zero. Partial coverage is not a completed baseline. Local and API latencies are different measurement conditions.

| Model | Benchmark | NLL | Brier | ECE (15 bins) | Median ms |
|---|---|---:|---:|---:|---:|
| jev | BBH | 0.3285 | 0.1140 | 0.0276 | 102.4 |
| jev | Belebele | 0.1917 | 0.0367 | 0.0198 | 99.9 |
| jev | Circa | 0.5908 | 0.2853 | 0.0870 | 95.2 |
| jev | ContractNLI | 1.1623 | 0.3117 | 0.1014 | 113.9 |
| jev | FinancialPhraseBank | 0.1563 | 0.0885 | 0.0371 | 101.6 |
| jev | JevBench_public_hard | 0.6200 | 0.3618 | 0.0877 | 109.1 |
| jev | JudgeBench | 0.4900 | 0.3099 | 0.0678 | 111.4 |
| jev | RAGTruth | 0.4401 | 0.2748 | 0.1068 | 112.8 |
| jev | TabFact | 0.3191 | 0.1956 | 0.0386 | 115.3 |
| jev | TruthfulQA_binary | 0.3213 | 0.1801 | 0.0850 | 97.8 |
| jev | WinoGrande | 0.3336 | 0.1247 | 0.0395 | 95.1 |
