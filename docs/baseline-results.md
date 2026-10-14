# Baselines — Jev run record

Jev completed the frozen first-pass panel on 2026-10-14. All 2,111 requests returned valid probabilities from jev-1.13.0. No training or calibration fitting was performed.

| Benchmark | n | Accuracy | NLL | Brier (sum) | ECE (15 bins) | Median API ms |
|---|---:|---:|---:|---:|---:|---:|
| BBH | 200 | 0.9251 | 0.3285 | 0.1140 | 0.0276 | 102.4 |
| Belebele | 200 | 0.9750 | 0.1917 | 0.0367 | 0.0198 | 99.9 |
| Circa | 200 | 0.7900 | 0.5908 | 0.2853 | 0.0870 | 95.2 |
| ContractNLI | 200 | 0.7800 | 1.1623 | 0.3117 | 0.1014 | 113.9 |
| FinancialPhraseBank | 200 | 0.9600 | 0.1563 | 0.0885 | 0.0371 | 101.6 |
| JevBench_public_hard | 111 | 0.7297 | 0.6200 | 0.3618 | 0.0877 | 109.1 |
| JudgeBench | 200 | 0.7850 | 0.4900 | 0.3099 | 0.0678 | 111.4 |
| RAGTruth | 200 | 0.8350 | 0.4401 | 0.2748 | 0.1068 | 112.8 |
| TabFact | 200 | 0.8650 | 0.3191 | 0.1956 | 0.0386 | 115.3 |
| TruthfulQA_binary | 200 | 0.8850 | 0.3213 | 0.1801 | 0.0850 | 97.8 |
| WinoGrande | 200 | 0.9100 | 0.3336 | 0.1247 | 0.0395 | 95.1 |

11-suite macro accuracy: **85.82%**. BBH accuracy is the equal-weight subtask average; other calibration metrics are example-weighted within each suite. ECE is descriptive and noisy at these sample sizes. API latency includes network. No confidence intervals or matched GPU/API latency claims.

The sampled FinancialPhraseBank all-agree subset is easier than lower-agreement configurations; its 96% score is not evidence of beating the vendor panel. Circa is a source-wide sample rather than a canonical test split. JevBench includes six ordinal Score decisions with argmax scoring.

Trace and config files: outputs/baselines-20261014/jev/. The config records runner SHA-256 and Git commit 909e47e. Subsequent GPU-only loader fixes do not alter these completed API predictions.

GPU status: waiting for Modal ac-worktrial authentication; no GPU jobs launched. PPLX/CLEF/CLEF-flash cells are pending, not measured zeroes.
