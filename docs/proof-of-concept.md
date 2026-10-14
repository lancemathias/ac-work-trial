# Decision model proof of concept

This is the working plan and evidence record for the first end-to-end decision
model experiment. Update it as choices are made and measurements arrive. The
research log holds individual experiments; this document captures the current
system, acceptance criteria and next steps.

**Status:** Planning. No implementation, training run or measured result is
represented by this document.

## Research question

Can a lightweight decision head trained on a frozen open source LLM improve
held-out classification accuracy and probability quality relative to the same
backbone without training, within one A100 and the available trial window?

## Scope and success criteria

| Component | Acceptance criterion | Evidence / current status |
| --- | --- | --- |
| Baseline | Reproducible untrained-backbone predictions with fixed prompts and label mappings | Pending |
| Training sanity check | Loss decreases and held-out performance improves on independent sanity-check data | Pending |
| Decision interface | Choice, Noul and ordered Score return valid outputs and probabilities | Pending |
| Training isolation | Evaluation datasets excluded from training and calibration; screening report retained | Pending |
| Evaluation | Per-benchmark results, denominators and explicit aggregation for all 11 benchmarks | Pending |
| Analysis | Failure traces motivate at least one controlled experiment | Pending |
| Serving | Measured warm/cold latency, throughput and memory on stated hardware | Pending |
| Reproduction | Saved head, immutable backbone revision, run config, data manifests and instructions | Pending |

Improvement must be measured against a matched baseline. A result that fails to
improve remains useful when the experiment isolates a clear hypothesis and the
failure is explained. Do not substitute a target score for an observed result.

## Initial model design

**Initial approach:** a strictly frozen causal LLM backbone and a small trainable
readout. Start with a linear head before testing a small MLP. Confirm the pooling
and option representation before extracting cached features.

**Follow-up step:** Consider base-model fine-tuning or LoRA after establishing
the frozen-backbone baseline and analyzing its failures. Adaptation should test
whether changing the representations improves on head-only training, with matched
data and evaluation and explicit compute accounting. Full fine-tuning is outside
the original trial brief and requires an agreed scope change before execution.

| Decision | Current choice | Rationale / evidence needed |
| --- | --- | --- |
| Backbone and revision | [To select] | Fit memory, preserve useful pretrained capabilities, establish baseline quality |
| Pooling | [To select] | Test whether the representation combines state, question and options effectively |
| Option scoring | [Fixed answer codes or shared candidate scorer] | Support variable option counts; document positional bias |
| Trainable parameters | [Count and names] | Freeze the backbone and verify optimizer scope |
| Context handling | [Token limit and rejection/chunking rule] | Avoid silent loss of decisive evidence |
| Calibration | [Temperature fitted on independent calibration data] | Measure probability quality without changing the predicted class |
| Bidirectional challenger | [Optional, only after baseline] | Separate checkpoint differences from claims about attention direction |

### Output contract

- **Choice:** distribution over supplied options, selected option and deterministic
  confidence derived from that distribution.
- **Noul:** probability that a statement is true; represent internally as two
  alternatives when convenient.
- **Score:** distribution over ordered rubric levels and its expected level index.
  The expected score can be continuous without a separate regression head.

Validate finite nonnegative probabilities, normalization, option membership and
rubric ordering. Match the intended API's confidence definitions. Confidence is a
summary of the distribution, not a separate guarantee of correctness. Keep
probability calibration distinct from schema validity.

## Data plan

Choose independent human-labelled task families that exercise sentiment, intent,
entailment, grounding and rubric judgments. Confirm sources, licenses, labels,
quality and overlap before inclusion. The toxic-comment dataset suggested in the
brief is a candidate for an initial training sanity check, subject to access and
license verification; it is not evidence of broad transfer.

| Partition | Purpose | Source / revision | Separation and quality checks |
| --- | --- | --- | --- |
| Training | Fit the decision head | [ ] | [ ] |
| Development | Choose recipe and analyze failures | [ ] | [ ] |
| Calibration | Fit probability temperature | [ ] | [ ] |
| Benchmark evaluation | Report transfer to the 11 named benchmarks | See research log | Never used as training or calibration examples |

Split related documents and generated templates together to reduce leakage.
Record transformations, option permutations, class balance, soft-label semantics,
duplicates, excluded items and provenance hashes.

## Infrastructure and artifacts

| Location | Contents | Required provenance |
| --- | --- | --- |
| Modal data Volume at /data | Source snapshots, processed datasets and frozen features | Dataset, tokenizer, prompt and backbone revisions; processing configuration |
| Modal model Volume at /models | Backbone and tokenizer caches | Immutable model revision and source |
| Modal results Volume at /results | Configs, metrics, traces, checkpoints and completion markers | Run ID, code revision, seeds, image/dependency versions and data hashes |
| Exported deliverables | Trained head, reproduction instructions, figures and research conclusions | Link back to the exact completed run |

Use a single serialized GPU execution path. Confirm A100 memory before model
selection. Record feature extraction separately from head training so cached
iteration times do not misrepresent end-to-end cost. Set explicit idle and
shutdown behavior within the authorized access window.

## First experiments

| Order | Question | Comparison | Stop / proceed rule |
| --- | --- | --- | --- |
| 1 | Is evaluation reproducible? | Repeat baseline on the same fixed manifest | Resolve inconsistent examples or metrics before training |
| 2 | Can the frozen representation support learning? | Untrained baseline versus linear head on independent sanity data | Investigate labels, features and optimization if learning fails |
| 3 | Does the head transfer? | Same head on held-out task families | Inspect failures before adding complexity |
| 4 | Is extra head capacity useful? | Linear versus small MLP, same features and data | Keep added capacity only with supported gains |
| 5 | Are returned probabilities useful? | Raw versus calibrated probabilities | Compare NLL, Brier and selective risk on independent held-out data |
| Follow-up | Does backbone adaptation resolve limitations of the frozen model? | Best frozen-backbone head versus LoRA or base-model fine-tuning with matched data/evaluation | Consider after failure analysis; confirm compute feasibility and scope before execution |

For each run, fill in the [research log](research-log.md) before and after
execution. Record negative results and changes to the protocol.

## Observability and failure analysis

Each evaluation trace should identify the example and source, prompt version,
question type, options, target, predicted distribution, selected answer, token
length, truncation/rejection status, latency and run ID. Aggregate errors by
benchmark, task family, input length, option count and confidence band.

Use representative traces to distinguish missing knowledge, reasoning failure,
ambiguous labels, option order sensitivity, missing evidence and data-processing
errors. Record sample counts before interpreting a pattern. Keep benchmark
inspection and subsequent training choices visible in the research log.

## Results and current conclusions

| Claim | Supporting run / artifact | Limitation |
| --- | --- | --- |
| [Fill only after measurement] | [ ] | [ ] |

## Open decisions and next actions

- [ ] Confirm access-window start/end, A100 memory and Modal environment.
- [ ] Pin benchmark subsets, conversions and reference aggregation.
- [ ] Select backbone and independent training/development/calibration sources.
- [ ] Freeze initial hypothesis, baseline and acceptance criteria.
- [ ] Record first run and inspect its failure traces.
- [ ] Export the final artifacts and select self-explanatory presentation figures.
