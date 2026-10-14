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

**Scope decision — October 14 2026:** Every backbone stays strictly frozen.
Full fine-tuning, LoRA and other backbone adapters are out of scope. Only the
new decision head and a separately fitted calibration temperature may be learned.
Synthetic data generation and synthetic training corpora are out of scope; use
public data with existing human labels and deterministic typed-decision conversions.
Public human-judged model responses are eligible without generating new responses.

**Initial approach:** start with a joint schema head, then compare a conventional
feed-forward head and a pointer head. Begin on Qwen3.5-9B and compare with
Qwen3.8-27B. Keep these as two checkpoint comparisons, not a pure parameter-count
ablation: the model generations differ as well as their sizes.

| Head | Proposed decision mechanism | Hypothesis and diagnostic |
| --- | --- | --- |
| Joint schema head, first | Trainable transformer routes frozen state evidence to questions/options and scores valid alternatives | Richer evidence routing helps grounding and complex schemas; inspect single-question versus joint requests for interference |
| Feed-forward head | Linear sanity control, then a small MLP on a fixed decision representation, with invalid option slots masked | Frozen representations already carry enough task information; extra routing adds little relative to its cost |
| Pointer head | Trainable projections score frozen option representations against a frozen question/decision representation | Explicit option representations improve variable-cardinality transfer and option-order robustness |

Use the same examples, label semantics, state/question content and evaluation
requests across heads. Record unavoidable layout/pooling differences, head
parameter counts, training steps and compute. A performance difference between
these complete heads does not isolate capacity from representation layout.
Do not describe a jointly trained head on frozen features as a reproduction of
Clef's adapter-trained checkpoint.

| Decision | Current choice | Rationale / evidence needed |
| --- | --- | --- |
| Initial backbones | Qwen3.5-9B; Qwen3.8-27B | Compare under matched head/data protocol; pin exact revisions before extraction |
| Head order | Joint schema, feed-forward, pointer | Test routing value, simpler readout sufficiency and option representation |
| Pooling and layout | Specify per head before feature extraction | Cache required final/question/option or token-level states without gold-label leakage |
| Trainable parameters | Head only; separate calibration temperature | Verify every backbone parameter frozen and unchanged |
| Context handling | Common supported token budget initially; record both tokenizer lengths | Reject or explicitly report overflow; no silent truncation or per-model denominator changes |
| Precision | Same where feasible; document quantization differences | Separate numerical/runtime effects from checkpoint differences |
| Attention | Native causal baseline; controlled noncausal attention experiment if valid | Changing a full-attention mask may leave other backbone operations causal |
| Expansion backbones | Gemma 4, all available sizes; Muse Glimmer; Nemotron, including eligible MoE checkpoints | Conditional on measured backbone effect and compute budget; verify exact IDs and availability first |

### Measuring whether backbone choice matters

Train the same joint-head design separately on each frozen Qwen backbone using
the same public training mixture, development/calibration partitions and examples.
Adapt input projections to hidden width, but keep head depth/width and training
budget comparable. Compare each trained model with its own untrained baseline,
and compare the trained models directly. Check head convergence before attributing
a weak result to the backbone. Repeat promising comparisons over three head seeds
when affordable; do not run a large model sweep before this comparison.

Primary comparison: macro accuracy over independent development task families.
Also report task-level accuracy, NLL, Brier, selective risk/coverage, peak memory,
feature extraction cost and warm end-to-end latency at matched request lengths.
Use paired, group-aware bootstrap intervals on identical examples; questions from
one source document stay together. Seed variation is a separate uncertainty source.
Report raw and separately calibrated probabilities on untouched development-test
items; fit temperatures only on the calibration partition.

**Provisional expansion gate, fixed before measurements:** a backbone advantage
of at least 1 percentage point in task-macro accuracy with a paired 95 percent
interval excluding zero and consistent direction across repeated seeds, or a
material calibration/serving advantage at comparable accuracy. Before running,
fill in the calibration and latency thresholds and available GPU-hour budget.
If the result is inconclusive, diagnose power, optimization and task slices rather
than declaring backbones equivalent. A practically useful domain-specific gain
can justify a targeted challenger, but record that new hypothesis before running.

If the gate passes, shortlist Gemma 4 sizes, Muse Glimmer and Nemotron from verified
checkpoints. Screen memory and feature-extraction cost first, then compare a small
representative set. Expand to all available Gemma 4 sizes only while the budget
supports it. For MoE, record total and active parameters separately: all resident
weights count toward memory even when only a subset is activated per token.

### Attention direction experiment

Hypothesis: bidirectional context improves question-conditioned evidence
representations, especially when decisive context follows a token being pooled.
Hold checkpoint weights, head architecture, source examples, prompt content,
precision and optimization fixed; refit an identical head separately for native
causal and changed-attention features. Cache keys must include the attention mode.
Report head-only convergence and zero-shot degradation as well as final accuracy.

First audit the backbone's full-attention, recurrent/linear-attention and
convolution paths. Qwen's hybrid layout means removing a triangular mask alone
may not make the entire model bidirectional. Label such a run a full-attention-mask
ablation, not a fully bidirectional encoder. If a valid frozen bidirectional path
requires retraining backbone weights or an unsupported kernel rewrite, mark that
variant infeasible under this scope. Do not combine attention changes with a new
backbone and attribute the result solely to bidirectionality.

### Output contract

**Decision — October 14 2026:** Use a per-question softmax over valid option
logits for Choice, Noul and ordered Score. Existing open decision models,
including Clef and pplx-decider, use this mapping with strong reported benchmark
results, making it a practical starting point. This is supporting precedent,
not an ablation proving softmax is optimal or evidence of Jev's private internals.
Sources: [Clef implementation](https://huggingface.co/Cloudflare/clef/blob/0b331204bb13fbd2ca93a64df1956f5b55478ce5/joint_schema_model.py)
and [pplx-decider implementation](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b/blob/d8c3bd18e31dc1804433d38528b3e770019bf00c/source/src/autojev/model.py).

Mask invalid option slots before normalization. Return the full distribution;
Noul returns its true probability, Choice selects the maximum, and Score returns
the probability-weighted level index. Softmax guarantees normalization, not
calibration. Keep the loss function, ordinal treatment and calibration temperature
as separate experiment choices; no alternative probability mapping is planned
initially. Choice denotes one selected alternative, not multi-label applicability.

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

Split related documents and source families together to reduce leakage.
Record deterministic transformations, option permutations, class balance,
existing human soft-label semantics, duplicates, exclusions and provenance hashes.
Do not generate synthetic examples, teacher training labels or model-authored distractors.
LLM judging is now authorized for auditing existing public data; optional algorithmic
checks and human adjudication may augment the audit. Persist original labels, judge
verdicts and any accepted corrections separately. Dataset sources remain unspecified.
Convert approved records to typed schemas with pinned Nimble converters where
available, then persist inputs, labels and provenance in the Modal data Volume.

## Infrastructure and artifacts

See the [training system diagram](training-system-design.md) for the public-data
audit, Nimble conversion, Modal storage and frozen-head training flow.

| Location | Contents | Required provenance |
| --- | --- | --- |
| Modal data Volume at /data | Source snapshots, processed datasets and frozen features | Dataset, tokenizer, prompt and backbone revisions; processing configuration |
| Modal test Volume at /test | Frozen benchmark inputs and separate gold labels; not mounted by training workers | Source/split/conversion manifest, IDs, groups and hashes |
| Modal model Volume at /models | Backbone and tokenizer caches | Immutable model revision and source |
| Modal results Volume at /results | Configs, metrics, traces, checkpoints and completion markers | Run ID, code revision, seeds, image/dependency versions and data hashes |
| Exported deliverables | Trained head, reproduction instructions, figures and research conclusions | Link back to the exact completed run |

Use a single serialized GPU execution path. Confirm A100 memory before model
selection. Record feature extraction separately from head training so cached
iteration times do not misrepresent end-to-end cost. Set explicit idle and
shutdown behavior within the authorized access window.

## Training recipe and hypotheses

Start with supervised cross-entropy over valid alternatives and AdamW for head
parameters only. Noul uses two alternatives; Score uses ordered levels and
existing label distributions where genuinely available. Do not fabricate soft
labels. The first trained experiment (E1) uses post-training temperature scaling:
fit a positive scalar temperature on the independent calibration partition by
minimizing NLL, then report both raw and calibrated probabilities on separate
development evaluation items.

Initial settings to preregister: learning rate 1e-3, weight decay 0.01, 10 percent
warm-up, gradient clipping at 1, effective batch 64 decisions, and a maximum of
five passes with independent-development early stopping. These are provisional
head-training defaults, not a vendor recipe or evidence of optimality. Choose
actual batch/sequence limits after measuring the feature-cache and memory costs.
Aim for roughly ten-minute cached-head experiments where feasible; token-level
joint heads can require larger caches and longer passes.

| Lever | Hypothesis | Controlled change and evidence |
| --- | --- | --- |
| Data balance | Large easy datasets dominate gradients and harm rare task transfer | Proportional versus task-balanced sampling at the same number of training decisions; compare per-family regressions |
| Label uncertainty | Hard majority labels overstate certainty on disputed cases | Existing human vote distributions versus hard labels on identical eligible rows; compare NLL/Brier and accuracy |
| Option presentation | The head learns answer positions rather than option meaning | Fixed versus deterministically permuted option order, preserving labels; evaluate paired order sensitivity |
| Probability objective | Cross-entropy alone leaves excessive certainty | Cross-entropy versus cross-entropy plus 0.1 Brier on the same examples; report raw and calibrated metrics |
| Learning rate | Head underfitting or instability limits the architecture comparison | Change learning rate only when convergence curves support the hypothesis; hold data and steps fixed |
| Training mixture | Missing task-family coverage explains observed failures | Add existing public human-labelled data from an independent source; hold total decisions fixed to separate coverage from scale |

Follow E1 with an RLCD-style experiment to tune the head's logits while keeping
the backbone frozen. Before running it, specify the reward/objective, update
algorithm and data partition; compare against E1's temperature-scaled head with
matched data and evaluation. Tune logits on training data, reserve calibration
data for temperature fitting, and keep benchmark test data excluded from both.
Report raw and temperature-scaled results for the follow-up to distinguish logit
training gains from calibration gains. RLCD here is a planned research direction,
not a claim to reproduce Jev's unpublished method; a supervised scoring loss
alone should not be described as reinforcement learning.

## Experiment sequence

| Stage | Comparison | Question / proceed rule |
| --- | --- | --- |
| E0 | Fixed evaluation manifests and repeatable untrained baselines | Resolve coverage, conversions and scoring before model selection |
| E1 | Joint schema head on frozen Qwen3.5-9B versus its untrained baseline; fit post-training temperature scaling | Validate learning and transfer; report raw and calibrated probabilities |
| E1 follow-up | RLCD-style head-only logit tuning versus the E1 temperature-scaled baseline | Preregister objective and update algorithm; isolate gains from logit tuning and temperature scaling |
| E2 | Joint head on Qwen3.5-9B versus Qwen3.8-27B, and each own baseline | Apply the backbone expansion gate on independent development data |
| E3 | Joint versus feed-forward versus pointer on a selected common backbone | Determine whether routing/option representations justify added cost |
| E4 | Native causal versus feasible attention-mask ablation, separately trained identical heads | Test evidence visibility; report hybrid causal operations that remain |
| E5 | One data/recipe change selected from trace evidence | Preregister the predicted affected slice and unchanged controls |
| Conditional | Verified Gemma 4 sizes, Muse Glimmer, Nemotron/MoE | Run only after the backbone gate or a documented targeted hypothesis |
| Final | Frozen chosen checkpoint/head and fixed calibration on all 11 benchmarks | Retain protocol differences and benchmark-guided selection disclosures |

The stages specify comparison dependencies, not a full Cartesian product of
backbones, heads, attention modes and recipes. Use independent development data
for routine selection. The brief permits exploratory benchmark evaluation and
trace analysis, but log every benchmark access and any choices it influences;
never fit head weights or temperature on benchmark examples or paraphrases.
Record each hypothesis and result in the [research log](research-log.md).

## Observability and failure analysis

**Tooling decision — 2026-10-14:** W&B is the single source of experiment
tracking. Modal provides execution logs and resource diagnostics. W&B tracks
training loss, learning rate, gradient norms, throughput, validation
metrics, run configuration and checkpoint references. JSONL is the authoritative
per-example trace format and records artifact metadata. Persist traces and actual
artifacts in the results Volume; model weights remain binary checkpoint files,
not JSONL. Link every record through run ID, checkpoint ID and example ID.

Modal Volumes persist working artifacts during runs. Keep durable results outside
Modal: version control code, configs, seeds, traces, metrics and documents; export
large tensor artifacts locally with verified checksums before cleanup. Never revert
research work when cleaning Modal. JSONL and local exports are durable evidence
copies, not a separate experiment tracker.
Future Modal operations use workspace `ac-worktrials`, profile `ac-worktrial`,
and environment `worktrial-lance`; the profile is not the environment name.
The W&B API key is configured in the primary checkout's ignored `.env`; it is
not present in this worktree. Inject only the W&B key through a Modal Secret
when launching training. Never include credentials in images, logs, configs,
JSONL traces or exported artifacts.

E1 code, input contract, launch/export commands and validation are documented in
[first-experiment training](first-experiment-training.md). Training data remains
unspecified; adding the loop does not launch a GPU job.

Phoenix is out of scope. Keep evaluation traces in JSONL with actual start/end
timestamps; use W&B for training instrumentation and Modal for resource metrics.
Include source hashes, artifact paths, completion status and preprocessing/model
revisions in the run manifest; periodically flush and commit durable results.

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

- [ ] Implement the test Volume mount boundary and fail-closed data hygiene gate described in [training system design](training-system-design.md); verify leak/manifest rejection before training.

- [ ] Deferred TODO: cache frozen-backbone token hidden states if repeated head experiments justify it. First measure one representative shard's size and read throughput; run full extraction overnight only if needed. No extraction is scheduled or launched by this TODO.
- [ ] Confirm access-window start/end, A100 memory and Modal environment.
- [ ] Pin benchmark subsets, conversions and reference aggregation.
- [ ] Pin Qwen3.5-9B and Qwen3.8-27B; select independent public human-labelled training/development/calibration sources.
- [ ] Specify joint, feed-forward and pointer layouts and their feature-cache needs.
- [ ] Fill in backbone expansion thresholds before measurements.
- [ ] Audit attention-mask feasibility for the chosen hybrid backbone.
- [ ] Freeze initial hypothesis, baseline and acceptance criteria.
- [ ] Record first run and inspect its failure traces.
- [ ] Export the final artifacts and select self-explanatory presentation figures.
