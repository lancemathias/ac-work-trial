# Decision models with frozen Qwen backbones

Applied Compute work trial · October 14–15, 2026 · Research handoff

## Background research — October 14

The task was to train a decision model on one A100, evaluate it on the eleven
benchmarks named by PPLX decider v1, and explain the methodology. We prioritized
reliable measurement, explicit data provenance and experiments that could explain
failure. Backbone fine-tuning, LoRA and synthetic training data were excluded.

We compared Jev's typed decision interface, PPLX's selected answer-token linear
readout, and Clef's joint schema head. Choice, Boolean Noul and ordered Score can
all be represented as distributions over permitted alternatives. Confidence does
not require a separate neural network: it can be computed from that distribution.
Ordinary labelled examples suffice to fit a scalar calibration temperature.

The released PPLX v1 implementation projects the final hidden state through a
bias-free 5,120 × 255 readout, initialized from pretrained answer-token weights:
1,305,600 parameters. Its training code permits backbone updates, so its published
quality is not evidence for equivalent frozen-backbone performance. Clef routes
option representations to evidence and jointly processes fields. Its released
backbone is also adapted. We reused its architecture and code, not those adapted
weights, for the first trainable-head experiments. Sources and architecture
comparisons are retained in [the original log](research-log.md); exact layers,
activations, shapes and precision are in [the architecture specification](joint-head-architecture.md).

## Establishing the evaluation contract — October 14

The eleven public sources yielded 65,377 normalized decisions. We fixed a
label-blind panel of 200 examples per suite, except all 111 public-hard JevBench
examples: **2,111 decisions**. BBH is averaged over its included subtasks first;
the final composite weights each of eleven suites equally. A rejected or missing
prediction counts as wrong in the primary score. Probability metrics are reported
on valid predictions, with their denominators retained. The common token ceiling
is 8,192, without silent truncation; several model protocols reject one long
ContractNLI item.

Source and preprocessing choices matter. We used FinancialPhraseBank's unanimous
subset, a GPT-4o JudgeBench cohort, English Belebele, fine-grained Circa labels,
response-level RAGTruth hallucination labels, and binary classification of
TruthfulQA MC2 candidate answers. These are declared conversions, not verified
reproductions of the vendor's unspecified panel or API prompts. Circa's source-wide
pool is designated evaluation-only here. Group IDs preserve shared contracts,
tables, questions and source contexts for conditional paired analysis.

The exact panel SHA-256 is
`50889a8285d85fc80ded905ed3534ec75cfe1e80d5dad84d8a02ae0150e90627`.
Dataset URLs, immutable revisions, conversion details and hashes are in
[`benchmark-preprocessing.json`](../data/manifests/benchmark-preprocessing.json)
and [`baseline-panel.json`](../data/manifests/baseline-panel.json).

We measured Jev 1.13, PPLX v1, Clef and Clef Flash under this common protocol.
Their coverage-adjusted composites were 85.82%, 81.82%, 82.16% and 76.58%.
Local GPU timings and hosted API timings are different measurement conditions;
we do not present them as a controlled serving-speed comparison. The fixed-panel
vendor baselines are reference measurements, not claims to reproduce model-card
scores. [Baseline protocol](baselines.md).

## Measuring initialization before training — October 14

Three independently initialized joint heads used seeds 20261014, 20261015 and
20261016. Original Qwen3.8-27B stayed frozen. Each head used width 1,024, two
routing layers, four joint field layers, 16 attention heads and FFN width 4,096:
128,056,324 parameters. CPU FP32 initialization and checkpoint storage preceded
BF16 inference. We computed backbone features once per example and applied all
three heads; we did not ensemble probabilities.

The average composite was **44.59% ± 1.36 percentage points**, sample SD across
initializations. These are not pure chance classifiers: Clef retains a frozen
pretrained lexical prior and deterministic scalar initializations. A later matched
9B control used E1's precision and initialization protocol; its per-seed results
are retained separately. This establishes initialization sensitivity, not the
capability of Qwen's original pretrained answer-token readout.
[27B protocol](random-joint-baseline.md), [9B control](random-joint-9b-baseline.md).

## E1: training the original joint head — October 14

E1 froze Qwen3.5-9B and trained a fresh 121,762,820-parameter head. The first
public-data release contained 5,000 records / 5,500 decisions. Connected families
and identical states were grouped before deterministic train/dev/calibration
assignment. There were 4,101 training records, 514 dev decisions and 497
calibration decisions. Labels, provenance and transformation traces were excluded
from model inputs by an explicit adapter whitelist.

The recipe used AdamW, LR 1e-3, weight decay 0.01, 10% warmup, clipping at 1,
effective batches of at least 64 decisions, and up to five epochs with early
stopping. The frozen backbone was BF16; head weights, gradients and optimizer
state were FP32, with BF16 autocast for eligible head operations and FP32 loss.
The accepted checkpoint was epoch 2 / step 142, selected by dev NLL. Its dev
accuracy improved from 32.49% before training to 66.93%. A separate calibration
split fitted T=0.7951; positive temperature does not change the winning class.

The benchmark composite was **57.35%**. The head learned, but transfer was uneven.
JudgeBench chose B on 197/200 examples; RAGTruth predicted no hallucination on
all 200; sentiment never predicted negative. The gold distributions did not
justify those collapses. Training loss continued falling while dev NLL worsened
after epoch 2, supporting generalization concerns without establishing one cause.
The initial schema adapter put option meanings in instructions and generic IDs
in option spans; benchmark requests often placed meanings directly in options.
That supplied a concrete representation-mismatch hypothesis.

## E2 and E3: normalizing inputs, scaling data and caching — October 14

E2 standardized training requests toward the benchmark-shaped state/question/
criteria representation and used a revised v2 public-data release (2,553 records).
The original-head result reached **60.19%**. Dev-only head ablations measured
73.66% for the original, 73.28% for Medium and 68.70% for Small. Medium changed
optimizer settings too, so this is not an isolated causal estimate of head size.

Medium used width 512, one evidence-routing layer, two joint field layers,
eight attention heads and FFN width 2,048. It has 25,206,788 parameters on 9B
and 28,354,564 on 27B. E3 used this head, LR 3e-4, five epochs and an expanded
v3 release of 7,928 records: ANLI, CICERO, CaseHOLD, HelpSteer2, Implicatures,
JEEBench, MultiRC, RACE and ReClor. Its grouped partitions supplied 6,262 training,
809 dev and 857 calibration decisions. E3 reached **59.85%**, so more data did
not establish a general improvement. Multiple factors changed from E1; these
historical comparisons cannot attribute gains to a single intervention.

Caching detached frozen hidden states removed repeated backbone work during
head optimization. Cache identity includes model/tokenizer revision, input bytes,
attention layout, precision and numerical runtime; each shard is checksummed.
Resumption required replay agreement, not merely matching a directory name.
An E1 cached control and cache-parity checks separated efficiency changes from
new scientific conditions. [Training system](training-system-design.md).

## Autoresearch: larger backbone and explicit selection — overnight October 14–15

The following work is explicitly **autoresearch**. Related operational changes
are grouped here; all registered hypothesis cards, unsuccessful attempts and
exact reproduction evidence remain in [the overnight log](overnight-research-log.md).
There were 45 registered cards at its recorded snapshot, including prepared-only
conditions; that count must not be interpreted as 45 completed model evaluations.

E4 switched the Medium head to frozen Qwen3.8-27B with v3 data. The user then
specified dev-accuracy checkpoint selection, earliest epoch on ties, rather than
NLL. We recovered the accuracy-selected saved checkpoint without retraining and
fit temperature separately. The fallback-runtime E4 composite was **60.93%**.
Its historical E3 contrast changes both backbone and selection, so the difference
is not a clean backbone-size effect. Every run retains its own accepted checkpoint
and recorded selection rule; later rules do not rewrite earlier selections.

## Autoresearch: faster kernels required a new numerical condition — October 15

Combined pinned causal-convolution and linear-attention kernels gave approximately
1.55–1.57× warm four-record backbone throughput in the measured workloads. They
failed the strict numerical-equivalence gate, despite agreement on sampled
argmax decisions in the combined condition. We therefore treated this as a new
runtime, regenerated features and retained the negative equivalence finding.
Optimized-runtime E4 reached **61.26%**. This is distinct from the 60.93%
fallback result. Kernel gains do not accelerate head training that already uses
cached features. JIT/cold-start cost and end-to-end extraction are separate costs.
[Measured performance and provenance](autoresearch-results.md).

## Autoresearch: exposing suffix evidence — October 15

E5 removed the full-attention causal mask while preserving padding isolation and
all weights. Qwen's recurrent/convolution paths remained causal: this is **not a
fully bidirectional encoder**. E5 used a separately extracted cache with the same
v3 recipe and optimized numerical runtime as its E4 control.

Its accepted epoch-2 checkpoint reached **68.41%**, versus 61.26% for E4, a
7.15-point observed gain. Finance and judging improved, while Circa and RAGTruth
regressed. This is a single-seed exploratory result. Paired example intervals
condition on fitted models and do not capture training variability.
[Attention experiment](autoresearch-e5-results.md).

## Autoresearch: repeatability before source attribution — October 15

Identical-seed fused-attention training controls diverged, eventually producing
58.60% versus 60.85% composites without a source intervention. Matching seeds,
initial weights and batch windows was insufficient. An actual 32-update head
probe found divergence with the automatic attention backward backend, while the
math backend matched parameters, gradients and losses exactly. The math fixture
ran at 0.844× the automatic backend's throughput. Full matched math controls
subsequently reproduced before the source screen continued.

This operational finding matters scientifically: an uncontrolled numerical
runtime could mimic a data gain. We retained both divergent outcomes instead of
selecting a favorable repeat. [Repeatability evidence](../outputs/autoresearch/head-determinism-summary.json).

## Autoresearch: public-source additions and a semantic repair — October 15

A matched 9B Medium-head screen used 6,512 training decisions, 510 updates,
fixed heldouts and one seed. Each condition added 250 decisions; the control added
fresh examples from existing sources. Source conditions and composite scores were:

| Condition | Composite | Difference from matched control |
| --- | ---: | ---: |
| Existing-source control | 60.93% | — |
| RuleTaker | 60.93% | +0.01 pp |
| Aegis2 | 60.20% | −0.73 pp |
| CivilComments | 61.45% | +0.52 pp |
| SNLI + MultiNLI | 61.05% | +0.12 pp |
| MASSIVE | 62.04% | +1.11 pp |
| BANKING77 | 60.38% | −0.55 pp |
| RuleTaker with explicit closed-world instructions | 60.52% | −0.42 pp versus original RuleTaker |

MASSIVE passed its screen thresholds but lacks independent seed confirmation.
Most additions did not resolve the targeted collapse. RuleTaker's closed-world
semantics motivated an instruction repair; the failed result was retained rather
than presented as a fix. Source-control results are not direct 27B comparisons.
[Source ledger](source-screen-ledger.md), [contrast verification](source-screen-round1.md).

## Autoresearch: recovering the pretrained readout — October 15

A simpler control kept original Qwen3.8-27B weights and projected its final prompt
state onto pretrained output rows for answer letters. There was no head training,
no generated explanation and no backbone adaptation. Only a scalar temperature
was fitted on the 857 nonbenchmark calibration examples. The native readout
reached **77.14%**, materially above the trained joint head.

A fixed two-pass recipe presented native and reversed option orders, realigned
both distributions to semantic IDs and averaged uncalibrated probabilities.
A separate temperature was fitted after pooling. It passed the preregistered
nonbenchmark dev gate and reached **79.49%** on the observed panel. This is the
strongest verified completed recipe in the indexed ledger, still below Jev's
85.82%. A dev-selected equal E5/native mixture reached 75.24% and was rejected.
The result suggests preserving pretrained answer semantics is a valuable baseline;
it does not isolate architecture from prompt and representation changes.
[Exact readout design](native-readout-design.md).

## Bounded thinking and residual-readout continuation — October 15

A fixed 128-token bounded-thinking protocol was separately preregistered, with
calibration first, a nonbenchmark dev gate, and a test-budget check. Its W&B run
is marked finished, but the audited local ledger lacks a sealed complete test
result. Partial logits or a passed dev gate are not a completed benchmark score;
we do not promote it from dashboard state alone.

The currently active user-directed continuation trains a 5,515,519-parameter
residual MLP: LayerNorm(5120), Linear(5120,1024), GELU, Linear(1024,255), added
to frozen pretrained code logits. It compares fixed option order with deterministic
per-epoch shuffling, preserving semantic targets. This differs from the joint-head
experiments and retains its original preregistration, selection and initialization.
It remains pending in this handoff snapshot. No running jobs were stopped or
changed for cleanup. [Residual protocol](residual-readout-v3.md).

## What the results establish

We trained and preserved several frozen-backbone decision heads, established a
common evaluation panel, investigated representation and data effects, and found
a useful partial-attention intervention. The strongest completed inference recipe
uses pretrained readout and option-order pooling, not a newly trained head. The
strongest completed trained-head condition in this ledger is E5 at 68.41%.
Both should be delivered, with their different purposes explicit.

Repeated observation of the panel makes results exploratory; it is not an untouched
final test. Conditional paired intervals do not establish cross-seed robustness.
Most trained conditions use one seed; only random initialization controls have
three-seed benchmark repeats. No claim of beating Jev or reproducing vendor
full-benchmark measurements is supported. Follow-up priorities are accepted-run
verification, unchanged independent dev selection, and targeted tests of semantic
binding; prepared conditions are not experimental successes.

## Reproduction and accepted checkpoint policy

The authoritative tracker is [W&B](https://wandb.ai/lmathias/ac-worktrial-decisions).
Historical benchmark imports are explicitly labelled retrospective; no inference
was rerun. Git retains code and reasoning. Modal retains versioned data, accepted
heads, other saved checkpoints and evaluation artifacts. Verified local exports
remain an independent copy outside Modal. The [Modal archive receipt](../outputs/handoff/modal-archive.json)
records the archived source and evaluation evidence; [preserved dataset receipts](../outputs/handoff/preserved-datasets.json)
identify content-addressed copies of previously uncommitted dataset files.

Use workspace `ac-worktrials`, profile `ac-worktrial`, environment `worktrial-lance`.
The [accepted-checkpoint catalog](accepted-checkpoints.md) and
[run index](../outputs/handoff/run-index.json) list **only the accepted
checkpoint for each run**, its hash, selection epoch and artifact path. Other
saved checkpoints remain reachable through the indexed original run directory;
we neither delete nor individually promote them. For native readout there is no
trained head checkpoint: the accepted artifact is a pinned backbone plus prompt,
pooling and temperature protocol. Its protocol digest is not a weight-file hash.

Reproduce the main reported measurements from archived predictions before
allocating a GPU:

```bash
python scripts/preprocess_benchmarks.py
python scripts/report_overnight_benchmarks.py
```

Use the original committed dataset release/config, not a newly sampled release.
For E1, see `docs/first-experiment-training.md`; E2/E3 configs are
`configs/e2-qwen35-9b-v2-cache.json` and `configs/e3-medium-v3.json`.
For E4/E5 use their exact native/bidirectional cache identities, selected accuracy
configs and kernel receipts. `scripts/export_training.py` checks complete remote
artifact inventories and hashes. A clean-checkout report requires restoring its
checksummed prediction exports; see `docs/reproduce-handoff.md` for explicit
commands. Do not launch the legacy parallel-baseline command by default.

### Model and dataset pins

| Component | Immutable revision / SHA-256 |
| --- | --- |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Clef head source | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |
| PPLX decider v1 | `d8c3bd18e31dc1804433d38528b3e770019bf00c` |
| Clef Flash | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` |
| E1 typed 5,000 | `3b3d8db82da7595565ad48c8cb7da981821097324c151169cf64c4c11fc3ea36` |
| v2 normalized 2,553 | `338a3584e896a17b011155c066387bb396879dd4a8f539aaf2c86a94fd257fd8` |
| v3 normalized 7,928 | `ef15c46f49192cafe5d35a1b0567e1a33cc13807fb5c2ca4d50e1fdbc6beec81` |

Full public-source pins are in `configs/training-source-pins.json` and the
benchmark manifest. Dependency sets are `requirements-training.txt` and
`requirements-kernels.txt`; exact runtime versions, code hashes and GPU metadata
remain in individual run configs. Dataset sources and usage terms remain attached
to their manifests. Frozen feature caches are optional reproducible acceleration,
not substitutes for the original inputs and model pins.

## Codebase map and major commits

The handoff branch consolidates evidence; `main` receives the selected delivery
path only through a PR. Experiments remain on their named branches. Unlisted
intermediate commits are implementation history, not additional experiments.
The complete [branch inventory](../outputs/handoff/run-index.json) records tips.

| Location / branch | Purpose |
| --- | --- |
| `main` | Published delivery code and documentation; update through PR |
| `deliverables/research-handoff` | New narrative, audit indexes, presentation and consolidation |
| `autoresearch/*` | Explicit autoresearch; grouped in the chronological sections above |
| `e1-review-fixes` | Historical E1 training and review hardening |
| `preservation/20261015-*` | Archival worktree snapshots; no new scientific result |
| `preservation/20261015-cacc` (`72568d5`) | Preserved dataset preparation/conversion code and provenance; dataset text stored separately in Modal |
| `scripts/preprocess_benchmarks.py`, `data/manifests/` | Fixed evaluation conversion and provenance |
| `scripts/train_joint.py`, `scripts/train_cached_joint.py` | Head training; frozen extraction or verified cached features |
| `scripts/pretrained_readout.py`, `scripts/readout_order_pool.py` | Pretrained answer-letter readout and semantic probability pooling |
| `scripts/report_overnight_benchmarks.py`, `outputs/traces/` | Verified reporting and compressed evaluation traces |
| `configs/`, `outputs/training/`, `outputs/test/` | Experiment inputs, accepted-checkpoint receipts and results |
| `outputs/model-artifacts/` | Ignored local tensor exports; paired Modal paths/hashes in index |
| `work/` | Ignored scratch and independent worktrees; never the only durable artifact copy |

Major historical milestones:

| Commit | Change |
| --- | --- |
| `7588454` | Initial research documents |
| `025aca3` | Four-model sampled baseline and calibration/cost audit |
| `bf364b7` | Three-seed frozen 27B joint-head baseline |
| `9a72a35` | E1-matched three-seed 9B random baseline |
| `0d321e2` | Completed E2 training and locked evaluation |
| `589524a` | Published E3 benchmark comparison |
| `9e77b62` | Accuracy-selected E4 recovery |
| `8e1b40c` | Adopted optimized numerical runtime |
| `af59dc3` | Verified suffix influence under the attention intervention |
| `02096dd` | Full math-backend repeatability gate and source-screen continuation |
| `67db897` | Negative RuleTaker semantic-repair result |
| `6593ae1` | Passed 27B option-order pooling dev gate |
| `2e7a58c` | Verified 79.49% recipe and bounded-thinking budget amendment |
| `366fe8a` | User-directed residual readout option-order experiment |
| `f1876e3`, `7260ce1` | Preserved handoff materials and consolidated experiment history |
| `66fe758` | New narrative, accepted inference package, presentation and verified checkpoint index |
| `9f27bc7` | Focused delivery branch prepared from published main; promotion requires PR |

The failed and prepared-only experiments remain documented in the overnight
journal and performance ledger. W&B state alone does not certify an accepted
checkpoint or completed test. The handoff validator checks artifacts and result
arithmetic; ongoing work is reconciled only from sealed evidence.
