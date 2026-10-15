# E2 validation comparison

Same 262-example E2 validation split; no fitting or calibration of vendor models on this split. CLEF-flash uses the pinned released weights; Jev uses API model jev-1.13.0. E2 uses its validation-selected epoch-4 checkpoint and T=1.51914063735723 fitted on separate calibration data. E2 validation was used for checkpoint selection, so it is not an unbiased final test comparison.

| Model | Valid / expected | Accuracy (misses wrong) | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Jev 1.13.0 | 262/262 | 87.40% | 0.4677 | 0.1668 | 0.0397 |
| CLEF-flash | 262/262 | 81.30% | 0.4957 | 0.2788 | 0.0605 |
| E2 calibrated | 262/262 | 73.66% | 0.6570 | 0.3869 | 0.0732 |

| Source | n | CLEF-flash | Jev 1.13.0 | E2 calibrated |
|---|---:|---:|---:|---:|
| ANLI | 43 | 88.37% | 88.37% | 79.07% |
| HelpSteer2 | 53 | 67.92% | 77.36% | 73.58% |
| MultiRC | 53 | 75.47% | 83.02% | 66.04% |
| RACE | 57 | 89.47% | 92.98% | 77.19% |
| ReClor | 56 | 85.71% | 94.64% | 73.21% |

Overall metrics weight each validation decision equally; source rows are descriptive, not a source-macro aggregate. NLL/Brier/ECE use valid outputs; accuracy counts missing outputs as wrong. NLL uses a 1e-12 floor. No vendor temperature fitting was performed.
