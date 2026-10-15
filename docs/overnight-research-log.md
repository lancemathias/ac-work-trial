# Overnight autoresearch issues hypotheses and fixes

Updated 2026-10-15T17:41:39.793660+00:00. Covers the October 14–15 overnight work and its morning continuations.

**Current best verified agent: 79.4905% composite; Jev 1.13 target: 85.82%.** The running thinking experiment has no completed accuracy result in this snapshot.

This log covers all 45 preregistered experiment/change cards, grouped into related entries, plus seven explicit operational failure receipts. Completed runs, rejected candidates, infrastructure work and prepared-only conditions are labeled separately. A targeted fix is not evidence that its proposed root cause was established.

All comparisons use the fixed 11-benchmark protocol. Entire benchmark datasets/tasks are excluded from training, dev and calibration. Checkpoint selection uses dev accuracy with earliest-epoch ties; temperature uses calibration only. Repeated panel comparisons are exploratory. Code and this log are versioned on an `autoresearch/` branch; tensor exports and checksum receipts remain outside Modal.

## User requested closeout

The user requested that the active thinking run be the final run and that the GPU be returned afterward. No follow-up candidate evaluations or GPU launches are authorized. Post-run keepalive is disabled. Closeout waits for the existing call, exports its completed results, stops its exact Modal app, and verifies that the target container is gone. Persistent volumes are retained.

## Earlier experiments reviewed

| Work | Issue and hypothesis | Observed result |
| --- | --- | --- |
| E1 | Original frozen 9B trained-head baseline. | 57.35% composite; RAGTruth already collapsed, so later data removal cannot explain its origin. |
| E2 | Revised data/recipe; review head-size ablations. | 60.19% composite. Dev original 73.66%, Medium 73.28%, Small 68.70%; Medium also changes optimizer, so not a clean size-only causal result. |
| E3 | Expanded v3 data with Medium head. | 59.85%, −0.34 pp versus E2; Boolean/Circa/NLI collapse persisted. |

These were reviewed as historical evidence rather than silently rerun overnight. [Original comparisons](autoresearch-results.md).

## Experiments and research infrastructure

### 1. Larger frozen backbone

- **Issue:** E3 regressed against E2 and retained task collapse.
- **Hypothesis:** A frozen 27B backbone may transfer better than 9B with the same head recipe.
- **Root cause or targeted fix:** Replace only the frozen backbone and extract fresh inputs; retain the matched recipe. A larger backbone is a proposed remedy, not an established cause of the regressions.
- **Outcome:** Accuracy-selected fallback E4: 60.9322% composite. Optimized-runtime E4: 61.2611%. Historical E3: 59.8528%; that historical contrast also changes selection/runtime.
- **Evidence:** [autoresearch-results.md](../docs/autoresearch-results.md).
- **Registered changes:** `e4`.

### 2. Remove the full attention causal mask

- **Issue:** A causal backbone can limit interactions among evidence, question and options.
- **Hypothesis:** Bidirectional full-attention layers improve evidence binding.
- **Root cause or targeted fix:** Remove causality only in full-attention blocks; recurrent and convolution paths remain causal.
- **Outcome:** Completed: 68.4141% versus matched native 61.2611%, +7.1530 pp. Circa and RAGTruth regress. Single seed; broad causal/seed robustness remains unconfirmed.
- **Evidence:** [autoresearch-e5-results.md](../docs/autoresearch-e5-results.md).
- **Registered changes:** `e5`.

### 3. Checkpoint selection by accuracy

- **Issue:** The original selection used dev NLL despite an accuracy-based objective.
- **Hypothesis:** Selecting maximum dev accuracy better matches the graded objective.
- **Root cause or targeted fix:** Use dev accuracy and earliest-epoch ties. Recover saved E4 epoch 5 without retraining; fit temperature separately on calibration.
- **Outcome:** Implemented and tested. Original NLL-selected results retained as historical; no selection by NLL going forward.
- **Evidence:** [autoresearch-results.md](../docs/autoresearch-results.md), [test_checkpoint_selection.py](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/tests/test_checkpoint_selection.py).
- **Registered changes:** `accuracy-checkpoint-selection`.

### 4. RuleTaker add-back

- **Issue:** Boolean entailment and negation transfer remain weak.
- **Hypothesis:** Explicit rule supervision may restore supported/unsupported distinctions.
- **Root cause or targeted fix:** Add 250 native-labelled RuleTaker decisions with normalized requests.
- **Outcome:** 60.9343%, +0.0062 pp versus 60.9281% equal-volume control; no demonstrated gain; TabFact declines 2.5 pp.
- **Evidence:** [ruletaker-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/ruletaker-decision.json), [source-screen-ledger.md](../docs/source-screen-ledger.md).
- **Registered changes:** `source-addback-ruletaker`.

### 5. Aegis2 add-back

- **Issue:** Boolean correctness training may not teach question-dependent polarity.
- **Hypothesis:** True=unsafe safety judgments may broaden Boolean transfer.
- **Root cause or targeted fix:** Add 250 Aegis2 decisions, preserving source labels.
- **Outcome:** 60.1975%, -0.7306 pp; targeted Boolean mean declines 4.25 pp; not advanced.
- **Evidence:** [aegis2-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/aegis2-decision.json), [source-screen-ledger.md](../docs/source-screen-ledger.md).
- **Registered changes:** `source-addback-aegis2`.

### 6. CivilComments add-back

- **Issue:** Boolean transfer may benefit from a second non-correctness domain.
- **Hypothesis:** Toxicity supervision may improve polarity generalization.
- **Root cause or targeted fix:** Add 250 CivilComments decisions; retain the native class imbalance.
- **Outcome:** 61.4523%, +0.5242 pp; targeted mean +0.75 pp; neither advance threshold passes.
- **Evidence:** [civilcomments-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/civilcomments-decision.json), [source-screen-ledger.md](../docs/source-screen-ledger.md).
- **Registered changes:** `source-addback-civilcomments`.

### 7. SNLI and MultiNLI add-back

- **Issue:** ContractNLI nearly omits Contradiction.
- **Hypothesis:** Cross-domain three-way NLI may restore contradiction predictions.
- **Root cause or targeted fix:** Add 250 decisions from the two NLI sources.
- **Outcome:** 61.0467%, +0.1186 pp; ContractNLI unchanged and Contradiction still absent; not advanced.
- **Evidence:** [snli-multinli-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/snli-multinli-decision.json), [source-screen-ledger.md](../docs/source-screen-ledger.md).
- **Registered changes:** `source-addback-snli-multinli`.

### 8. MASSIVE add-back

- **Issue:** Circa favors a dominant option; many-option training coverage was reduced.
- **Hypothesis:** Intent classification may improve competing-option discrimination.
- **Root cause or targeted fix:** Add 250 MASSIVE decisions with the same counts, updates and heldouts.
- **Outcome:** 62.0386%, +1.1105 pp; Circa +6.5 pp. Both screen criteria pass, but the conditional interval includes zero; new-seed confirmation required.
- **Evidence:** [massive-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/massive-decision.json), [source-screen-ledger.md](../docs/source-screen-ledger.md).
- **Registered changes:** `source-addback-massive`.

### 9. BANKING77 add-back

- **Issue:** The same many-option hypothesis might transfer from banking intents.
- **Hypothesis:** More competing intent options may reduce option bias.
- **Root cause or targeted fix:** Add 250 BANKING77 decisions independently of MASSIVE.
- **Outcome:** 60.3814%, -0.5467 pp; Circa -4.5 pp; not advanced.
- **Evidence:** [banking77-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/banking77-decision.json), [source-screen-ledger.md](../docs/source-screen-ledger.md).
- **Registered changes:** `source-addback-banking77`.

### 10. Equal-volume source control

- **Issue:** A larger training set or extra updates could masquerade as a useful restored source.
- **Hypothesis:** Fresh examples from existing domains provide a fair volume control.
- **Root cause or targeted fix:** Add 250 unused existing-source decisions; match 6,512 train, 809 dev, 857 calibration, 510 updates and seed.
- **Outcome:** Completed control: 60.9281%. This is the comparator for all six source screens.
- **Evidence:** [source-screen-ledger.md](../docs/source-screen-ledger.md), [math-source-full-repeat-proof.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/math-source-full-repeat-proof.json).
- **Registered changes:** `source-addback-control`.

### 11. RuleTaker instruction repair

- **Issue:** The normalized RuleTaker question omitted native closed-world semantics.
- **Hypothesis:** Making negation-as-failure explicit may repair a supervision/instruction mismatch.
- **Root cause or targeted fix:** Confirmed instruction omission: state the closed-world rule explicitly while retaining the exact 250 records, native labels and training windows; extract fresh features for changed requests.
- **Outcome:** Completed: 60.5166%, -0.4177 pp versus original RuleTaker. TabFact +0.5 pp; both thresholds fail. The omission was real, but is not established as the cause of benchmark regressions.
- **Evidence:** [ruletaker-cwa-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/ruletaker-cwa-decision.json), [ruletaker-cwa-actual-contrast.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/ruletaker-cwa-actual-contrast.json).
- **Registered changes:** `ruletaker-explicit-closed-world`.

### 12. Frozen pretrained answer readout

- **Issue:** Small supervised heads transfer poorly and collapse on some suites.
- **Hypothesis:** The frozen backbone may already know the decisions better than the trained head.
- **Root cause or targeted fix:** Use pretrained answer-letter output rows, no head training, and one calibration-only temperature.
- **Outcome:** Completed: 77.1359% composite, 82.4475% dev; stronger than the trained heads. This does not prove which training source caused their failures.
- **Evidence:** [readout-direct-result.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/readout-direct-result.json), [readout-results.md](../docs/readout-results.md).
- **Registered changes:** `pretrained-answer-readout-control`, `pretrained-readout-optimized-runtime-trial`.

### 13. Single-token answer codes

- **Issue:** Numeric answer codes become multiple tokens for larger option sets.
- **Hypothesis:** Uppercase letters can cover all panel options with one token each.
- **Root cause or targeted fix:** Confirmed tokenizer issue: replace numeric codes with A–Z; normalize tokenizer outputs to flat IDs; check every encoded input.
- **Outcome:** Implemented before candidate accuracy measurement; full input/tokenizer checks passed.
- **Evidence:** [readout-tokenizer-letters-v1.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/readout-tokenizer-letters-v1.json).
- **Registered changes:** `pretrained-readout-single-letter-codes`.

### 14. Boolean intercept calibration

- **Issue:** Native Boolean predictions may have a transferable True/False offset.
- **Hypothesis:** A single calibration-fitted intercept may improve accuracy.
- **Root cause or targeted fix:** Fit a fixed-grid intercept on 137 permitted Boolean calibration decisions; require the preregistered overall dev gain.
- **Outcome:** Bias +0.875 improves Boolean dev accuracy, but overall gain is only +0.9889 pp, below +1 pp. Rejected; no candidate test evaluation or threshold relaxation.
- **Evidence:** [readout-boolean-offset-dev.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/readout-boolean-offset-dev.json).
- **Registered changes:** `readout-boolean-offset-calibration`.

### 15. Reverse option order probability pool

- **Issue:** Answer-letter or display position may influence decisions.
- **Hypothesis:** Pooling native and reversed displays may reduce position sensitivity.
- **Root cause or targeted fix:** Align semantic option IDs and average the two uncalibrated probability vectors equally; fit global T on calibration only. Position sensitivity is a proposed mechanism.
- **Outcome:** Completed and promoted: 79.4905% versus 77.1359%, +2.3546 pp; dev 84.3016%. TabFact +10 pp and RAGTruth +4.5 pp. Current verified best; Jev target 85.82% remains unmet.
- **Evidence:** [readout-order-decision.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/readout-order-decision.json), [readout-results.md](../docs/readout-results.md).
- **Registered changes:** `pretrained-readout-reversed-order-pooling`.

### 16. Equal trained-head and pretrained mixture

- **Issue:** Two agents might have complementary errors.
- **Hypothesis:** A fixed equal mixture may beat the stronger individual model.
- **Root cause or targeted fix:** Mix cached E5 and pretrained probabilities equally; fit a separate calibration-only T after the fixed dev gate.
- **Outcome:** Dev gate passes, but test is 75.2376%, below native 77.1359%. Rejected. Higher dev accuracy did not guarantee transfer.
- **Evidence:** [cached-mixture-result-receipt.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/cached-mixture-result-receipt.json), [cached-mixture-vs-readout-paired.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/cached-mixture-vs-readout-paired.json).
- **Registered changes:** `cached-e5-pretrained-equal-mixture`.

### 17. Bounded native thinking

- **Issue:** Direct readout may omit useful reasoning.
- **Hypothesis:** A fixed short scratchpad may improve evidence and reasoning decisions.
- **Root cause or targeted fix:** Enable native thinking with 128 tokens; keep answer probabilities from the frozen readout. Template overhead required a capacity-only amendment; original input content was preserved. Budget calibration/dev and test separately with unchanged safety factors and accuracy gates.
- **Outcome:** GPU smoke capacity passed at about 64.2 GB peak. Cold/warm measurements retained. Calibration completed (857/857); dev inference is running (568/809 observed 2026-10-15T17:44:25.015614+00:00). No completed dev accuracy or thinking test result yet.
- **Evidence:** [thinking-staged-budget-amendment.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/thinking-staged-budget-amendment.json), [thinking-staged-warm-result.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/thinking-staged-warm-result.json), [thinking-staged-trial-call.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/thinking-staged-trial-call.json).
- **Registered changes:** `pretrained-bounded-thinking128`.

### 18. Shortest-example few-shot selection

- **Issue:** A small fixed example set might teach polarity and entailment cheaply.
- **Hypothesis:** Seven short demonstrations may improve frozen-model decisions.
- **Root cause or targeted fix:** Select the shortest approved training record in each of seven source/label cells. Semantic audit found context-free Aegis prompts and questionable NLI inferences.
- **Outcome:** Rejected before any model inference. Original examples and CPU evidence preserved; native labels were not silently corrected.
- **Evidence:** [fewshot-seven-state.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/fewshot-seven-state.json).
- **Registered changes:** `pretrained-native-seven-fewshots`.

### 19. Audited seven-example prompt

- **Issue:** The shortest-only selection contained ambiguous teaching examples.
- **Hypothesis:** Clear self-contained demonstrations may teach the intended distinctions.
- **Root cause or targeted fix:** Lock seven original training examples using explicit semantic quality criteria and audit reasons; preserve source labels and query semantics.
- **Outcome:** Prepared, not GPU-tested. Exact membership and all 3,777 input fingerprints passed; 1,034 added tokens, maximum final prompt 9,203. Ten helper/evaluator tests and the next resident-service CPU checks passed. GPU smoke and accuracy trial pending.
- **Evidence:** [fewshot-quality-v2-selection.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/fewshot-quality-v2-selection.json), [fewshot-quality-v2-preparation.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/fewshot-quality-v2-preparation.json), [readout-candidates-worker-cpu.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/readout-candidates-worker-cpu.json).
- **Registered changes:** `pretrained-native-seven-fewshots-quality-v2`.

### 20. Selective thinking on option-order disagreement

- **Issue:** Full thinking is expensive and may damage already-correct pooled decisions.
- **Hypothesis:** Order disagreement may identify decisions that benefit from reasoning.
- **Root cause or targeted fix:** Preregistered before thinking accuracy: use the pool when semantic argmax agrees; otherwise use fixed128 thinking. No fitted routing threshold or suite routing.
- **Outcome:** Prepared only: routes 203/857 calibration and 180/809 dev decisions. Fourteen combined CPU checks include synthetic full scoring. Actual selective dev accuracy awaits complete thinking outputs; subset-only GPU path still pending if needed.
- **Evidence:** [selective-thinking-preparation.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/selective-thinking-preparation.json), [selective-cached-runtime-cpu.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/selective-cached-runtime-cpu.json).
- **Registered changes:** `pretrained-order-disagreement-thinking128`.

### 21. Optimized kernels and matched rebaseline

- **Issue:** Frozen-backbone iterations were slow; optimized kernels might change numerics.
- **Hypothesis:** Optimized convolution and linear-attention kernels may reduce iteration time.
- **Root cause or targeted fix:** Measure combined and isolated kernels on matched inputs. Confirmed BF16 numerical drift means fresh matched caches and controls, rather than claiming equivalence.
- **Outcome:** Combined speedup 1.55–1.57×, 64/64 sampled decisions agree, but strict numeric parity fails (maximum probability difference about 0.0438). Conv-only 1.04–1.05×, 62/64 agreement; FLA-only 1.43–1.45×, 64/64. Adopted as a new numerical regime, not a proven quality improvement.
- **Evidence:** [summary.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/optimized-kernels-v1/summary.json), [summary.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/optimized-kernels-isolation-v1-conv_only/summary.json), [summary.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/optimized-kernels-isolation-v1-linear_only/summary.json).
- **Registered changes:** `optimized-kernel-runtime`, `optimized-kernel-isolation`, `optimized-runtime-matched-rebaseline`, `optimized-runtime-adoption-decision`.

### 22. Warm-worker reuse

- **Issue:** Separate short-lived workers repeatedly loaded weights and rebuilt runtime state.
- **Hypothesis:** Keeping one backbone and tokenizer resident removes repeated setup.
- **Root cause or targeted fix:** Confirmed lifecycle overhead: one A100 worker, persistent weight volume, one in-memory pool, no blind retries, five-minute keepalive and reservation-bounded idle expiry.
- **Outcome:** GPU reuse proof passed. Current thinking smokes and inference use the same worker with one backbone load. Most repeated weight progress bars were volume-to-GPU reads rather than fresh Hub downloads.
- **Evidence:** [warm-worker-resident-proof.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/warm-worker-resident-proof.json), [thinking-staged-warm-result.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/thinking-staged-warm-result.json).
- **Registered changes:** `resident-single-gpu-worker`.

### 23. Cache composition and union filtering

- **Issue:** Repeated feature extraction and copied add-backs waste setup time.
- **Hypothesis:** Exact parent reuse and extraction of the union once can preserve inputs while saving work.
- **Root cause or targeted fix:** Validate parent manifests, request hashes, labels and lexical rows; compose or filter only selected source records. Extract changed RuleTaker requests afresh.
- **Outcome:** CPU composition contracts passed. Real source union was reused across the six screens; repaired RuleTaker used fresh changed-request features. No model-quality claim for caching itself.
- **Evidence:** [cache-composition-tests.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/cache-composition-tests.json), [source-fast-union-manifest.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/source-fast-union-manifest.json), [ruletaker-cwa-composed-cache.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/ruletaker-cwa-composed-cache.json).
- **Registered changes:** `cache-composition-runtime`, `addback-union-cache-filter`.

### 24. Interrupted E4 cache recovery

- **Issue:** Preemption and strict cache-identity guards blocked reuse of 7,232 already-extracted records.
- **Hypothesis:** Verified shards can be resumed without changing the scientific condition.
- **Root cause or targeted fix:** Confirmed replay/directory and metadata issues: preserve attempts; distinguish physical GPU UUID from computational identity; explicitly migrate only missing native-attention metadata; require first/last live feature parity and every shard checksum.
- **Outcome:** Recovery completed and E4 finished. The migration did not authorize arbitrary backend/model differences or mixing partial scientific results.
- **Evidence:** [e4-live-resume-parity.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/e4-live-resume-parity.json), [e4-resume-metadata-difference.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/e4-resume-metadata-difference.json), [cache-resume-legacy-cpu-tests.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/cache-resume-legacy-cpu-tests.json).
- **Registered changes:** `e4-preemption-recovery`, `e4-physical-gpu-resume`, `e4-legacy-native-metadata-migration`.

### 25. Identical-seed source controls diverged

- **Issue:** Same-seed runs differed enough to undermine one-point source claims.
- **Hypothesis:** Fused attention backward nondeterminism may amplify during BF16 head training.
- **Root cause or targeted fix:** Compare real 32-step replicas and autograd operations; use math SDPA only for the trainable head, then repeat two full 510-step controls. The exact individual nondeterministic operation is not conclusively isolated.
- **Outcome:** Auto replicas diverged; math replicas matched. Two complete math controls exactly match selected weights, all 510 updates, dev/cal artifacts, T and all 2,111 semantic responses; both 60.9281%. Same-worker/same-seed proof only. Short math probe throughput was about 15.6% lower.
- **Evidence:** [warm-control-repeat-divergence.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/warm-control-repeat-divergence.json), [head-determinism-probe-v2-result.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/head-determinism-probe-v2-result.json), [math-source-full-repeat-proof.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/math-source-full-repeat-proof.json).
- **Registered changes:** `warm-source-control-repeat`, `head-backward-determinism`, `full-math-source-control-confirmation`.

### 26. Paired reporting and collapse diagnostics

- **Issue:** A composite score can hide class collapse or sampling noise.
- **Hypothesis:** Paired group comparisons and per-class recall clarify gains and failures.
- **Root cause or targeted fix:** Rescore exported predictions, retain the fixed denominator, report suite/class distributions and paired native-group bootstrap intervals.
- **Outcome:** Implemented for completed contrasts. Intervals are conditional on reused samples and omit seed/method-selection uncertainty; no independent-confirmation claim.
- **Evidence:** [compare_predictions.py](../scripts/compare_predictions.py), [source-screen-ledger.md](../docs/source-screen-ledger.md), [confirmation-availability.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/confirmation-availability.json).
- **Registered changes:** `paired-comparison-reporting`.

### 27. Boolean label-matched controls

- **Issue:** An add-back gain might come from class proportions rather than source content.
- **Hypothesis:** A matched MultiRC addition can isolate class/type exposure.
- **Root cause or targeted fix:** Prepare 250-record controls matching RuleTaker, Aegis2 or CivilComments label counts; run only if a preregistered Boolean screen trigger passes.
- **Outcome:** Prepared and CPU-validated, not GPU-run. None of the Boolean source screens met its trigger.
- **Evidence:** [experiment-cards.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/experiment-cards.json).
- **Registered changes:** `boolean-type-label-controls`.

### 28. Polarity diagnostic

- **Issue:** Improved Boolean accuracy could merely shift the preferred class.
- **Hypothesis:** Correct/incorrect paraphrase pairs test sensitivity to question polarity.
- **Root cause or targeted fix:** Prepare 94 permitted dev pairs (188 requests), no fitting. Audit old runner paths and restore the saved math head backend before execution.
- **Outcome:** Prepared only; old runner needs adaptation to completed math attempts. No polarity model result claimed.
- **Evidence:** [polarity-math-adaptation-audit.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/polarity-math-adaptation-audit.json).
- **Registered changes:** `validation-polarity-diagnostic`.

### 29. ANLI Boolean views

- **Issue:** MultiRC is the sole explicit v3 Boolean source.
- **Hypothesis:** Explicit Boolean entailment views may transfer beyond replaying the same choices.
- **Root cause or targeted fix:** Prepare matched human-labelled ANLI Boolean views and choice-replay controls; retain original allowed source groups.
- **Outcome:** Prepared and validated, not GPU-run.
- **Evidence:** [experiment-cards.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/experiment-cards.json).
- **Registered changes:** `e6-candidate`.

### 30. Second-round source candidates

- **Issue:** Sentiment, evidence decisions, narratives, causality and ordinal rubric coverage remain possible gaps.
- **Hypothesis:** Each restored supervision type may improve its prespecified transfer diagnostic.
- **Root cause or targeted fix:** Prepare separate 250-decision SST2, PubMedQA, MCScript2+MCTest, WIQA and HelpSteer2 ordinal conditions. PubMedQA has 229 Yes/21 No/zero Maybe; it does not positively teach Maybe.
- **Outcome:** Preparation and tokenizer checks only, no GPU results. GSM8K was withheld after a label-independent distractor heuristic solved the proposed constructed examples; no misleading ablation run was launched.
- **Evidence:** [experiment-cards.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/experiment-cards.json), [gsm8k-distractor-audit.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/gsm8k-distractor-audit.json).
- **Registered changes:** `phase2-sst2`, `phase2-pubmedqa`, `phase2-narrative`, `phase2-wiqa`, `phase2-helpsteer2-ordinal-flat`.

### 31. New-seed confirmation

- **Issue:** E5 and MASSIVE gains could depend on one initialization/order.
- **Hypothesis:** Matched new seeds may reproduce the observed improvements.
- **Root cause or targeted fix:** Prepare paired seeds 20261015/20261016, fixing original data splits and recipes. MASSIVE requires a positive effect in each new seed and at least +1 pp mean; original screen is excluded from that confirmation mean.
- **Outcome:** Configs and CPU checks prepared; no new-seed GPU results yet. E5 and MASSIVE remain exploratory.
- **Evidence:** [attention-repeat-cpu.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/attention-repeat-cpu.json), [massive-confirmation-cpu.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/massive-confirmation-cpu.json).
- **Registered changes:** `e5-seed-confirmation`, `massive-matched-seed-confirmation`.

## Operational failures and corrections

These failures are retained even when a corrected attempt later passed. They are not successful model experiments.

### Warm worker startup import

- **Issue:** Warm worker startup import failed during setup or validation.
- **Hypothesis:** Correcting the diagnosed implementation/fixture issue should allow the unchanged scientific condition to run.
- **Root cause and targeted fix:** Remote sibling imports could not resolve /project/scripts. Restore the known script import path and run CPU checks before GPU handoff.
- **Outcome:** Stopped and corrected; no GPU allocated.
- **Evidence:** [warm-worker-cpu-failure-launch.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/warm-worker-cpu-failure-launch.json).

### Readout order startup import

- **Issue:** Readout order startup import failed during setup or validation.
- **Hypothesis:** Correcting the diagnosed implementation/fixture issue should allow the unchanged scientific condition to run.
- **Root cause and targeted fix:** Missing /project/scripts bootstrap caused ModuleNotFoundError at startup. Add the bootstrap before sibling imports; one corrected CPU attempt.
- **Outcome:** Crash watcher stopped the app; failure retained; subsequent CPU checks passed.
- **Evidence:** [readout-order-preparation-v1-failure.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/readout-order-preparation-v1-failure.json).

### Determinism probe replay

- **Issue:** Determinism probe replay failed during setup or validation.
- **Hypothesis:** Correcting the diagnosed implementation/fixture issue should allow the unchanged scientific condition to run.
- **Root cause and targeted fix:** Platform preemption replayed an input despite retries=0; an existing empty output directory failed mkdir. Immutable request identity, separate bounded attempt directories, idempotent completed-result checks, at most three attempts.
- **Outcome:** Original probe produced no valid comparison; corrected probe completed.
- **Evidence:** [head-determinism-probe-v1-failure.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/head-determinism-probe-v1-failure.json).

### RuleTaker release map ordering

- **Issue:** RuleTaker release map ordering failed during setup or validation.
- **Hypothesis:** Correcting the diagnosed implementation/fixture issue should allow the unchanged scientific condition to run.
- **Root cause and targeted fix:** Filesystem-dependent manifest key order changed serialized release bytes. Sort file-map paths before serialization; retain old metadata.
- **Outcome:** CPU guard failed before GPU work; corrected check passed with unchanged data/config.
- **Evidence:** [ruletaker-cwa-cpu-v1-failure.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/ruletaker-cwa-cpu-v1-failure.json).

### Persistent cache mount comparison

- **Issue:** Persistent cache mount comparison failed during setup or validation.
- **Hypothesis:** Correcting the diagnosed implementation/fixture issue should allow the unchanged scientific condition to run.
- **Root cause and targeted fix:** Resolved shard paths were compared against an unresolved mount anchor. Resolve both paths before containment checks.
- **Outcome:** Original CPU check failed before payload reads; corrected persistent-cache check passed.
- **Evidence:** [readout-cache-cpu-v1-failure.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/readout-cache-cpu-v1-failure.json).

### Thinking tokenizer preflight

- **Issue:** Thinking tokenizer preflight failed during setup or validation.
- **Hypothesis:** Correcting the diagnosed implementation/fixture issue should allow the unchanged scientific condition to run.
- **Root cause and targeted fix:** The default thinking template did not serialize identically to explicit no-thinking mode. Inspect actual pinned template modes, preserve public query content, fingerprint explicit thinking inputs and account only for template overhead.
- **Outcome:** CPU-only feasibility failure retained; amended preflight passed before generation.
- **Evidence:** [thinking-readout-preflight-v1-failure.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/thinking-readout-preflight-v1-failure.json).

### Thinking evaluator test fixture

- **Issue:** Thinking evaluator test fixture failed during setup or validation.
- **Hypothesis:** Correcting the diagnosed implementation/fixture issue should allow the unchanged scientific condition to run.
- **Root cause and targeted fix:** A synthetic pass-branch fixture actually produced a failed dev gate. Correct the fixture predictions to exercise a real pass; retain production gate behavior.
- **Outcome:** Fixture failure retained; corrected runtime checks passed. No production threshold was relaxed.
- **Evidence:** [thinking-smoke-runtime-cpu-v1-fixture-failure.json](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/thinking-smoke-runtime-cpu-v1-fixture-failure.json).

## Remaining diagnostic qualifications

- Calibration timing: reasoning accounts for 93.8% of measured generation-plus-readout compute; 587/857 requests hit the fixed128-token budget and270 closed earlier. These batch-amortized timings exclude loading/preparation/volume commits and do not establish an accuracy effect. [Timing metadata](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/thinking-calibration-timing.json).
- Exact Boolean ties differ between current dev and benchmark tie ordering. The read-only audit found one native dev tie and four pooled dev ties. Active gates were not retrospectively changed. [Metric audit](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/metrics-rule-audit.json).
- The old polarity runner needs completed math-attempt paths and restoration of the saved head backend before use. It has no measured model result.
- Crash-loop protection stops startup failures or repeated tracebacks. Platform preemption is handled with preserved bounded attempts and durable call IDs, not blind resubmission.
- Thinking GPU capacity and warmup succeeded, but the original upfront conservative full-run estimate did not fit. The scheduling-only amendment preserves numerical settings and thresholds and checks each phase against the reservation.

## Durable records and maintenance

- [Central performance ledger of every completed scored run, including control replicas and random-head seeds](performance-ledger.md).
- [Benchmark scores beside all four released baselines](overnight-benchmark-scores.md), including headline runs, source ablations and earlier context.
- [Machine-readable full log and original hypothesis/fix/success-criterion cards](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/overnight-research-log.json).
- [Original preregistration registry](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/experiment-cards.json), including pinned settings, seeds and success criteria.
- [Matched source results](source-screen-ledger.md), [readout results](readout-results.md), and [pending readout experiments](readout-next-experiments.md).
- [Current execution handoff](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/continuation.json), [live-status observation](https://github.com/lancemathias/ac-work-trial/blob/deliverables/research-handoff/outputs/autoresearch/overnight-log-live-status.json), and [versioned trace index](../outputs/traces/index.json).

Append or update an entry when a condition finishes, fails, changes its targeted fix, or is explicitly rejected. Keep original hypotheses and failed outcomes; add a fresh dated result rather than rewriting the history. Prepared-only work must not be reported as an executed model experiment.
