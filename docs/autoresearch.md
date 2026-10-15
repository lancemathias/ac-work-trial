# Autonomous decision-agent research

Goal active. Policy: `outputs/autoresearch/policy.json`.

## E4 preregistration

Hypothesis: frozen Qwen3.8-27B features improve decision transfer over E3 Qwen3.5-9B, preserving v3 data, grouped splits, Medium head architecture, seed, optimizer, five epochs, dev-NLL selection and independent temperature calibration. Hidden input dimension changes with backbone. Primary observed outcome is the fixed 11-suite calibrated coverage-adjusted macro accuracy; report paired per-suite differences and prediction distributions. Improvement over E3 supports promotion; seed confirmation follows before claiming robust superiority. Do not change evaluation samples or drop failures.

Setup: full v3 frozen feature extraction is a one-off cost and may exceed ten minutes. Train/evaluation timings reported separately. No benchmark task enters train/dev/calibration. Audit all v3 records and source lineage before launch.

## Additional user instruction

All changes and tests must be principled, repeatable and scientific. Work on separate `autoresearch/*` branches. Hypotheses, pinned inputs, controlled comparisons, appropriate validation and retained negative results are mandatory. This policy governs the active goal.

## Attention-mask ablation (requested)

After E4, compare its native-causal backbone to the identical frozen Qwen3.8-27B with bidirectional full-attention masks. Preserve padding isolation, all weights, data and partition IDs, Medium head recipe and seed. Re-extract the entire feature cache; never reuse causal features under a noncausal label. Hypothesis: evidence after question/option tokens becomes available to their hidden representations, improving semantic binding and task transfer. Acceptance: higher fixed composite with per-suite/collapse diagnostics, plus confirmation across seeds for a promising effect. Inspect the pinned implementation first. Qwen hybrid recurrent paths may remain causal; explicitly report them and avoid claiming a fully bidirectional backbone. This is a post-training attention-layout change to a frozen model, so degradation is also plausible.

E5 implementation inspection confirms the pinned text model accepts independent full-attention/recurrent mask mappings. A forward pre-hook supplies a bidirectional valid-key mask to full attention and calls the native recurrent-mask helper unchanged. Full-attention `is_causal` flags are disabled; no tensor weights change. CPU behavioral tests demonstrate suffix influence, causal-prefix invariance in control, padding isolation and unchanged frozen parameters. All 19 targeted tests passed on 2026-10-14 (Pacific). E5 receives its own cache, and refuses a cache with a different attention mode.
