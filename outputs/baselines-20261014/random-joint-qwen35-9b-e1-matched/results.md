# E1-matched Qwen3.5-9B untrained-head baseline

Three independent head initializations; mean ± sample standard deviation across seeds. Seed 20261014 uses E1’s saved pre-training head. The frozen backbone, architecture, FP32 head parameters and BF16 autocast match E1. T=1; no training or calibration fitting.

Where coverage is incomplete, accuracy is valid-only / misses counted wrong. These are averages of scores, not an ensemble.

| Benchmark | n per seed | Accuracy | Valid predictions per seed |
|---|---:|---:|---|
| BBH | 200 | 35.29% ± 8.73 | [200, 200, 200] |
| Belebele | 200 | 22.17% ± 5.25 | [200, 200, 200] |
| Circa | 200 | 0.83% ± 0.29 | [200, 200, 200] |
| ContractNLI | 200 | 49.08% ± 2.53 / 48.83% ± 2.52 | [199, 199, 199] |
| FinancialPhraseBank | 200 | 35.83% ± 11.64 | [200, 200, 200] |
| JevBench_public_hard | 111 | 35.44% ± 4.96 | [111, 111, 111] |
| JudgeBench | 200 | 55.50% ± 1.50 | [200, 200, 200] |
| RAGTruth | 200 | 61.67% ± 7.11 | [200, 200, 200] |
| TabFact | 200 | 51.67% ± 1.26 | [200, 200, 200] |
| TruthfulQA_binary | 200 | 39.67% ± 3.25 | [200, 200, 200] |
| WinoGrande | 200 | 52.00% ± 1.00 | [200, 200, 200] |
| **11-suite macro** | **2111** | **39.92% ± 3.21 / 39.90% ± 3.21** | |

| Seed | Valid-only macro | Misses counted wrong |
|---|---:|---:|
| 20261014 | 42.97% | 42.95% |
| 20261015 | 40.22% | 40.20% |
| 20261016 | 36.57% | 36.54% |

GPU worker elapsed: 8.16 minutes; peak allocated memory: 20.84 GiB. All backbone parameter version counters remained unchanged.

Full calibration metrics and seed scores are in metrics.json; checkpoint provenance and hashes are in config.json, e1-reference.json and initial-head-checksums.json.
