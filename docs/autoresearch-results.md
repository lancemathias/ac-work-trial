# Autoresearch experiment journal

Primary metric: fixed 11-suite calibrated coverage-adjusted macro accuracy, averaging BBH subtasks before the suite macro. Panel repeatedly observed; no claim of an untouched holdout.

| Experiment | Hypothesis/change | Fixed composite | Status |
|---|---|---:|---|
| Jev 1.13.0 | Existing matched API baseline | 85.82% | Complete |
| E1 | Original 9B head training | 57.35% | Complete |
| E2 | Revised v2 data and recipe | 60.19% | Complete |
| E3 | Medium head, expanded v3 data | 59.85% | Complete; -0.34 pp versus E2, single seed |
| E4 | E3 recipe with frozen Qwen3.8-27B and fresh cached features | Pending | Extracting features; 17 CPU tests passed |
| E5 | E4 recipe, bidirectional full attention | Pending | Queued after E4; 19 CPU tests passed |

Validation comparisons: E2 original 73.66%, E2 Medium 73.28%, E2 Small 68.70%. E3 validation population differs; do not compare aggregates as matched estimates. Medium versus original changes optimizer jointly.

## E3 interpretation

Calibrated composite 59.8528%, versus E2 60.1925% (-0.3397 percentage points). E3 is not a demonstrated improvement. Single seed; no significance claim. Dev accuracy 73.6712%, NLL 0.607045, temperature 1.106902. Validation populations differ from E2. Training and test artifacts exported locally with checksum verification under `outputs/model-artifacts/`; lightweight receipts are versioned under `outputs/training/` and `outputs/test/`.

Persistent dominance: RAGTruth False 200/200 (gold False 132/200); TruthfulQA False 198/200 (gold False 127/200). Small BBH subtasks also have one-label outputs; treat those as diagnostic flags with small sample counts. Source audit finds 6,928 Choice and 1,000 Noul records in v3; all Noul comes from MultiRC, with 454 True / 546 False overall. E3 Noul dev accuracy is 50% on 94 records, versus Choice 76.78% on 715. Thus absence of Noul training or simple all-false label imbalance is not established. Broader binary entailment/evidence data is a candidate hypothesis, to be chosen after the controlled backbone/mask tests.
