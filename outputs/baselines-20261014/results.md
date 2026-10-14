# Baselines — measured results

| Benchmark | n | PPLX v1-27B | CLEF-flash | CLEF | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
| BBH | 200 | 81.46% / 81.46% (200/200) | 69.50% / 69.50% (200/200) | Pending | 92.51% / 92.51% (200/200) |
| Belebele | 200 | 96.50% / 96.50% (200/200) | 95.50% / 95.50% (200/200) | Pending | 97.50% / 97.50% (200/200) |
| Circa | 200 | 76.50% / 76.50% (200/200) | 77.50% / 77.50% (200/200) | Pending | 79.00% / 79.00% (200/200) |
| ContractNLI | 200 | 84.50% / 84.50% (200/200) | 85.43% / 85.00% (199/200) | Pending | 78.00% / 78.00% (200/200) |
| FinancialPhraseBank | 200 | 92.50% / 92.50% (200/200) | 94.00% / 94.00% (200/200) | Pending | 96.00% / 96.00% (200/200) |
| JevBench_public_hard | 111 | 72.07% / 72.07% (111/111) | 64.86% / 64.86% (111/111) | Pending | 72.97% / 72.97% (111/111) |
| JudgeBench | 200 | 77.50% / 77.50% (200/200) | 63.00% / 63.00% (200/200) | Pending | 78.50% / 78.50% (200/200) |
| RAGTruth | 200 | 77.50% / 77.50% (200/200) | 65.00% / 65.00% (200/200) | Pending | 83.50% / 83.50% (200/200) |
| TabFact | 200 | 90.50% / 90.50% (200/200) | 72.50% / 72.50% (200/200) | Pending | 86.50% / 86.50% (200/200) |
| TruthfulQA_binary | 200 | 68.00% / 68.00% (200/200) | 58.00% / 58.00% (200/200) | Pending | 88.50% / 88.50% (200/200) |
| WinoGrande | 200 | 83.00% / 83.00% (200/200) | 97.50% / 97.50% (200/200) | Pending | 91.00% / 91.00% (200/200) |
| **Macro: missed excluded** | **2111** | 81.82% | 76.62% | Pending | 85.82% |
| **Macro: missed wrong** | **2111** | 81.82% | 76.58% | Pending | 85.82% |

PPLX is perplexity-ai/pplx-decider-v1-27b (v1), pinned to d8c3bd18e31dc1804433d38528b3e770019bf00c. No v1.1 results are included.

Each cell is **accuracy excluding missed trials / accuracy counting missed trials as wrong**, followed by valid/expected coverage. A miss is any rejected, failed, or unattempted panel ID. A valid prediction disagreeing with gold counts as wrong in both versions. BBH uses its subtask macro-average in both versions. Macro rows are equally weighted across 11 benchmarks. Pending is not zero. Calibration is measured on valid predictions only. Local and API latencies are different measurement conditions.

| Model | Benchmark | NLL | Brier | ECE (15 bins) | Median ms |
|---|---|---:|---:|---:|---:|
| pplx | BBH | 0.4916 | 0.2611 | 0.1118 | 183.8 |
| pplx | Belebele | 0.1073 | 0.0433 | 0.0459 | 183.6 |
| pplx | Circa | 0.8048 | 0.3634 | 0.1834 | 178.6 |
| pplx | ContractNLI | 0.4331 | 0.2401 | 0.0434 | 956.6 |
| pplx | FinancialPhraseBank | 0.2373 | 0.1224 | 0.0839 | 156.5 |
| pplx | JevBench_public_hard | 0.5358 | 0.3115 | 0.0778 | 316.2 |
| pplx | JudgeBench | 0.4664 | 0.3042 | 0.0869 | 751.5 |
| pplx | RAGTruth | 0.4755 | 0.3088 | 0.0407 | 406.6 |
| pplx | TabFact | 0.2307 | 0.1327 | 0.0804 | 251.1 |
| pplx | TruthfulQA_binary | 0.5883 | 0.3985 | 0.1119 | 156.2 |
| pplx | WinoGrande | 0.3661 | 0.2310 | 0.0387 | 157.8 |
| clef-flash | BBH | 0.6828 | 0.3667 | 0.0756 | 103.0 |
| clef-flash | Belebele | 0.1533 | 0.0622 | 0.0355 | 104.7 |
| clef-flash | Circa | 0.8772 | 0.3777 | 0.1681 | 109.6 |
| clef-flash | ContractNLI | 0.3858 | 0.2065 | 0.0606 | 360.0 |
| clef-flash | FinancialPhraseBank | 0.1906 | 0.0920 | 0.0538 | 92.0 |
| clef-flash | JevBench_public_hard | 0.8050 | 0.4675 | 0.1201 | 146.0 |
| clef-flash | JudgeBench | 0.7252 | 0.4960 | 0.1701 | 277.5 |
| clef-flash | RAGTruth | 1.1167 | 0.6281 | 0.3006 | 165.3 |
| clef-flash | TabFact | 0.6189 | 0.4092 | 0.1175 | 131.5 |
| clef-flash | TruthfulQA_binary | 0.8822 | 0.5929 | 0.2567 | 92.3 |
| clef-flash | WinoGrande | 0.0958 | 0.0454 | 0.0207 | 92.6 |
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
