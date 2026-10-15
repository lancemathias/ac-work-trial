# Central performance ledger

[Native readout design and terminology](native-readout-design.md).

Updated 2026-10-15T22:53:45.801180+00:00. **30 completed scored runs**: 20 research evaluations including repeated controls, four released baselines, and six individual random-head seed baselines.

All values are accuracy percentages on the same fixed2,111-example/11-benchmark panel. Misses count as wrong; BBH averages subtasks first. Composite is the equal mean of11benchmark scores. E1–E3 and random-head runs are historical context, not newly rerun overnight.

## Every completed benchmark run beside the baselines

| Run | Composite | Delta vs Jev pp | BBH | Belebele | Circa | ContractNLI | Finance | JevBench hard | JudgeBench | RAGTruth | TabFact | TruthfulQA | WinoGrande |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.13 | 85.82 | +0.00 | 92.51 | 97.50 | 79.00 | 78.00 | 96.00 | 72.97 | 78.50 | 83.50 | 86.50 | 88.50 | 91.00 |
| PPLX v1 | 81.82 | -4.00 | 81.46 | 96.50 | 76.50 | 84.50 | 92.50 | 72.07 | 77.50 | 77.50 | 90.50 | 68.00 | 83.00 |
| CLEF | 82.16 | -3.66 | 75.97 | 96.00 | 74.50 | 84.00 | 92.50 | 74.77 | 75.00 | 76.50 | 85.00 | 74.50 | 95.00 |
| CLEF Flash | 76.58 | -9.24 | 69.50 | 95.50 | 77.50 | 85.00 | 94.00 | 64.86 | 63.00 | 65.00 | 72.50 | 58.00 | 97.50 |
| Native + reverse | 79.49 | -6.33 | 75.42 | 96.00 | 77.50 | 80.50 | 81.50 | 72.97 | 77.50 | 76.50 | 83.00 | 72.50 | 81.00 |
| Native 27B | 77.14 | -8.68 | 75.42 | 96.50 | 73.50 | 79.00 | 83.00 | 72.07 | 76.00 | 72.00 | 73.00 | 69.50 | 78.50 |
| Joint-head + native mixture | 75.24 | -10.58 | 72.04 | 94.00 | 59.50 | 82.00 | 85.50 | 72.07 | 80.00 | 62.50 | 68.50 | 72.00 | 79.50 |
| Bidirectional full attention | 68.41 | -17.40 | 69.69 | 93.50 | 50.00 | 64.50 | 86.00 | 69.37 | 78.00 | 41.00 | 63.50 | 59.00 | 78.00 |
| MASSIVE | 62.04 | -23.78 | 56.58 | 82.00 | 42.50 | 65.50 | 88.50 | 42.34 | 67.50 | 65.50 | 47.50 | 64.00 | 60.50 |
| CivilComments | 61.45 | -24.36 | 55.13 | 80.50 | 38.50 | 66.50 | 87.50 | 42.34 | 63.50 | 66.00 | 52.00 | 63.50 | 60.50 |
| E4 optimized | 61.26 | -24.56 | 61.11 | 88.50 | 60.50 | 51.50 | 43.50 | 61.26 | 69.50 | 61.50 | 55.50 | 46.50 | 74.50 |
| SNLI + MultiNLI | 61.05 | -24.77 | 54.17 | 81.50 | 35.50 | 65.50 | 85.50 | 46.85 | 64.50 | 66.00 | 47.50 | 64.00 | 60.50 |
| RuleTaker | 60.93 | -24.88 | 55.13 | 79.00 | 37.50 | 66.00 | 88.50 | 44.14 | 65.00 | 66.00 | 47.00 | 64.00 | 58.00 |
| E4 fallback | 60.93 | -24.88 | 58.70 | 89.00 | 62.00 | 52.00 | 47.00 | 58.56 | 70.50 | 62.50 | 51.50 | 45.50 | 73.00 |
| Source control | 60.93 | -24.89 | 56.46 | 81.00 | 36.00 | 65.50 | 81.00 | 47.75 | 63.00 | 66.00 | 49.50 | 64.50 | 59.50 |
| Math head control repeat | 60.93 | -24.89 | 56.46 | 81.00 | 36.00 | 65.50 | 81.00 | 47.75 | 63.00 | 66.00 | 49.50 | 64.50 | 59.50 |
| Auto head control 1 | 60.85 | -24.97 | 53.20 | 81.00 | 40.50 | 58.50 | 91.50 | 44.14 | 65.50 | 66.00 | 47.50 | 61.00 | 60.50 |
| RuleTaker repaired | 60.52 | -25.30 | 55.74 | 81.50 | 36.50 | 63.50 | 90.00 | 45.95 | 63.00 | 66.00 | 47.50 | 55.50 | 60.50 |
| BANKING77 | 60.38 | -25.44 | 55.25 | 81.50 | 31.50 | 67.50 | 88.50 | 41.44 | 65.50 | 65.50 | 47.00 | 60.50 | 60.00 |
| Aegis2 | 60.20 | -25.62 | 54.23 | 81.50 | 37.50 | 68.00 | 89.00 | 45.95 | 62.00 | 57.00 | 47.50 | 58.00 | 61.50 |
| E2 | 60.19 | -25.62 | 60.87 | 79.00 | 34.50 | 59.50 | 78.50 | 47.75 | 63.50 | 66.00 | 47.50 | 62.50 | 62.50 |
| E3 | 59.85 | -25.96 | 56.04 | 80.50 | 31.50 | 66.00 | 80.50 | 42.34 | 65.00 | 66.00 | 47.00 | 63.50 | 60.00 |
| Auto head control repeat | 58.60 | -27.21 | 52.78 | 80.00 | 30.50 | 58.00 | 86.00 | 42.34 | 64.00 | 63.50 | 47.50 | 58.50 | 61.50 |
| E1 | 57.35 | -28.46 | 55.86 | 81.00 | 61.00 | 53.00 | 48.00 | 45.05 | 44.50 | 66.00 | 56.50 | 60.00 | 60.00 |
| Random head 27B seed 20261015 | 45.79 | -40.03 | 28.50 | 34.50 | 0.50 | 39.50 | 61.00 | 44.14 | 59.50 | 66.00 | 53.50 | 63.50 | 53.00 |
| Random head 27B seed 20261016 | 44.87 | -40.95 | 27.42 | 30.50 | 3.50 | 52.50 | 55.00 | 39.64 | 55.00 | 66.00 | 52.50 | 57.00 | 54.50 |
| Random head 27B seed 20261014 | 43.11 | -42.71 | 35.08 | 22.50 | 0.50 | 36.00 | 56.00 | 39.64 | 45.50 | 66.00 | 52.50 | 63.50 | 57.00 |
| Random head 9B seed 20261014 | 42.95 | -42.87 | 44.20 | 27.50 | 1.00 | 48.50 | 44.00 | 38.74 | 55.50 | 65.00 | 53.00 | 43.00 | 52.00 |
| Random head 9B seed 20261015 | 40.20 | -45.61 | 34.90 | 22.00 | 0.50 | 46.50 | 41.00 | 37.84 | 54.00 | 66.50 | 51.50 | 36.50 | 51.00 |
| Random head 9B seed 20261016 | 36.54 | -49.27 | 26.75 | 17.00 | 1.00 | 51.50 | 22.50 | 29.73 | 57.00 | 53.50 | 50.50 | 39.50 | 53.00 |

## Attempted or prepared work without a complete test score

| Run or condition | Status |
| --- | --- |
| E2 Medium and Small head ablations | Dev-only records:73.28% and68.70%; no comparable completed11-benchmark test scores |
| Bounded thinking128 | Running final GPU run; no complete test score yet |
| Boolean offset | Rejected by devaccuracy gate; no test evaluation |
| Original shortest seven-shot prompt | Rejected during semantic audit before model inference |
| Audited seven-shot prompt | CPU prepared; deferred by user stop-after-current request |
| Selective disagreement thinking | CPU prepared; deferred by user stop-after-current request |
| MASSIVE new-seed pairs | Prepared; not GPU-run |
| Bidirectional full attention new-seed pairs | Prepared; not GPU-run |
| ANLI Boolean views and phase2 sources | Prepared; not GPU-run |
| Polarity diagnostic and label-matched controls | Prepared; not GPU-run |

Pending, rejected-before-test and interrupted work are not assigned zero or fabricated benchmark scores. All45 registered changes and operational failures are described in the [issues hypotheses and fixes log](overnight-research-log.md).

## Data and regeneration

- [CSV with one row per completed run](../outputs/autoresearch/performance-ledger.csv).
- [CSV with benchmarks as rows](../outputs/autoresearch/overnight-benchmark-scores.csv).
- [Machine-readable scores and source hashes](../outputs/autoresearch/overnight-benchmark-scores.json).
- [Readable grouped comparisons and definitions](overnight-benchmark-scores.md).

Regenerate with `python3 scripts/report_overnight_benchmarks.py`. The generator inventories completed local test directories and refuses mismatched panels, changed checked metrics, or inconsistent composite arithmetic. Early E1/E2 reports are rederived from checksum-verified predictions; repeated controls are retained as separate rows. The final thinking evaluation will be added after its complete export is available.

Headline gains and source effects are exploratory on a reused panel; the ledger does not establish independent confirmation or cross-seed robustness.
