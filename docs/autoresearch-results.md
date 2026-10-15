# Autoresearch experiment journal

Primary metric: fixed 11-suite calibrated coverage-adjusted macro accuracy, averaging BBH subtasks before the suite macro. Panel repeatedly observed; no claim of an untouched holdout.

| Experiment | Hypothesis/change | Fixed composite | Status |
|---|---|---:|---|
| Jev 1.13.0 | Existing matched API baseline | 85.82% | Complete |
| E1 | Original 9B head training | 57.35% | Complete |
| E2 | Revised v2 data and recipe | 60.19% | Complete |
| E3 | Medium head, expanded v3 data | Pending | Training complete; evaluation active |
| E4 | E3 recipe with frozen Qwen3.8-27B and fresh cached features | Pending | 17 CPU tests passed; waits for E3 |
| Next mask ablation | E4 recipe, bidirectional full attention | Pending | Requested; implementation inspection |

Validation comparisons: E2 original 73.66%, E2 Medium 73.28%, E2 Small 68.70%. E3 validation population differs; do not compare aggregates as matched estimates. Medium versus original changes optimizer jointly.
