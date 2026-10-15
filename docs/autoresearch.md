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

## Crash-loop policy

User requests prompt triage because prolonged loops notify their manager. A local watchdog checks registered research apps every 20 seconds and stops an app on its first container-startup failure or repeated unhandled traceback. CLI reads/stops have timeouts. It preserves failure logs and Modal volumes. Do not relaunch the same failing configuration blindly: diagnose, fix and pass targeted validation before a single new attempt. Application retries default to zero; platform startup/preemption recovery is separate, so the watchdog remains necessary. Session 24828 runs `scripts/watch_modal_crashes.py`; inspect `outputs/autoresearch/crash-watch.jsonl` and status. GPU reservation deadline remains in policy.

## Candidate E6: Boolean transfer versus matched replay

Motivation: E3 Boolean development accuracy is 50%, with near-constant False predictions on RAGTruth/TruthfulQA; existing Boolean labels are balanced enough that simple label imbalance is not an established cause. Hypothesis: applying existing human entailment/contradiction annotations through the Noul interface improves Boolean transfer. Select only original ANLI training rows, exclude neutral, and deterministically balance entailment/contradiction. Preserve state and family provenance. Two conditions add the same 414 selected parent decisions: (A) replay their original Choice rendering; (B) represent entailment as True and contradiction as False using a fixed Noul question. The replay controls extra examples, exposure and optimizer-update count. No novel contexts or inferred teacher labels. Dev and calibration files stay byte-identical to E3/E4.

Candidates are prepared, not launched. Choose the backbone/attention context only after E4/E5 results; keep it identical within this paired comparison. Preserve all other recipe settings, select on dev and fit temperature on independent calibration. Primary evidence: paired calibrated fixed composite and Boolean-suite prediction distributions; also inspect unchanged original Noul dev slice and all-suite regressions. A gain requires seed confirmation before a robust claim. Full-benchmark task exclusions inherit the approved parent dataset proof because states/sources are unchanged and new views come exclusively from verified original train rows.

## Mandatory change record (user instruction)

Every research change requires **Hypothesis** (what can improve and why), **Targeted change** (exact intervention; control and fixed factors), and **Success criterion** (measurable pre-result decision rule). Record outcome and keep/reject/inconclusive afterward. Structured cards: `outputs/autoresearch/experiment-cards.json`. E4/E5 use a practical threshold of +1 absolute percentage point on the fixed composite; E6 also requires +5 points on the unchanged Boolean dev slice to support its proposed mechanism. The numerical thresholds were clarified while E4 was extracting features, before its training/test outcome; E5/E6 have not trained. Earlier experiments already observed are retrospective evidence, never retroactively labelled preregistered. Runtime-only fixes use direct correctness/health checks as their success criterion.

Cache-composition runtime validation: 23 targeted CPU tests passed, including exact live-versus-composed head updates, exact replay features, and rejection of corrupted bytes or changed alias labels. The composer does not modify the training loader or head. Real 27B composite-cache integration remains pending until E4/E5 outcome and E6 launch decision.
