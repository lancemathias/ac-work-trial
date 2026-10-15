# Autonomous decision-agent research

Goal active. Policy: `outputs/autoresearch/policy.json`.

## E4 preregistration

Hypothesis: frozen Qwen3.8-27B features improve decision transfer over E3 Qwen3.5-9B, preserving v3 data, grouped splits, Medium head architecture, seed, optimizer, five epochs, dev-NLL selection and independent temperature calibration. Hidden input dimension changes with backbone. Primary observed outcome is the fixed 11-suite calibrated coverage-adjusted macro accuracy; report paired per-suite differences and prediction distributions. Improvement over E3 supports promotion; seed confirmation follows before claiming robust superiority. Do not change evaluation samples or drop failures.

Setup: full v3 frozen feature extraction is a one-off cost and may exceed ten minutes. Train/evaluation timings reported separately. No benchmark task enters train/dev/calibration. Audit all v3 records and source lineage before launch.

## Additional user instruction

All changes and tests must be principled, repeatable and scientific. Work on separate `autoresearch/*` branches. Hypotheses, pinned inputs, controlled comparisons, appropriate validation and retained negative results are mandatory. This policy governs the active goal.
