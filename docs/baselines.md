# Baselines — first matched decision-model panel

Date: 2026-10-14. This is an independent evaluation under explicit assumptions,
not a reproduction of Perplexity's undisclosed 11-benchmark panel. These models
are existing decision checkpoints, not untrained LLM backbones. No training,
temperature fitting, prompt optimization or checkpoint selection occurs here.

## Frozen inputs

The first pass uses 200 label-blind examples per suite, except JevBench public hard
(all 111). Total: 2,111 decisions shared by all four models. Selection is SHA-256
ranking of `20261014:<id>`. BBH is round-robin over included subtasks. Source
groups are retained for later cluster-aware analysis, but sampling is at the row
level; these are not 2,111 independent observations.

`data/manifests/baseline-panel.json` contains every selected ID, split, class
count, processed-file checksum and immutable open-model revision. Panel SHA-256:
`50889a8285d85fc80ded905ed3534ec75cfe1e80d5dad84d8a02ae0150e90627`.
`data/manifests/benchmark-preprocessing.json` contains upstream revisions/URLs,
raw-file and processed-file SHA-256 hashes, exclusions and full-data statistics.
The runner rejects altered processed datasets or preprocessing manifests.

| Benchmark | Assumption / conversion | Full processed n | First-pass n |
|---|---|---:|---:|
| WinoGrande | XL labeled validation; 2 choices | 1267 | 200 |
| FinancialPhraseBank | All-agree; positive/negative/neutral | 2264 | 200 |
| RAGTruth | Test responses; any annotated span means hallucination | 2700 | 200 |
| JudgeBench | GPT-4o cohort; select objectively better A/B answer | 350 | 200 |
| BBH | Explicit choices/Boolean tasks; exclude free-form and malformed rows | 5508 | 200 |
| JevBench public hard | All public hard items; native Choice/Noul/Score | 111 | 111 |
| TabFact | Full labeled test; statement entailed by supplied table | 12779 | 200 |
| ContractNLI | Test contracts; full document; 3-way NLI, no gold evidence | 2091 | 200 |
| Circa | Source-wide fine-grained majority; exclude NA; not a canonical test split | 31525 | 200 |
| Belebele | English only; 4 choices | 900 | 200 |
| TruthfulQA binary | Each MC2 candidate separately true/false; group by question | 5882 | 200 |

Canonical option order is preserved in this pass. JudgeBench is single ordering;
the two-order diagnostic is a separate follow-up, not silently averaged here.
BBH is averaged across included subtasks before entering the 11-suite equal-weight
macro-average. Every other suite uses example accuracy. Score hard predictions
use the most likely ordinal level, not rounded expectation; expected-level MAE
is additionally recorded. See benchmark-preprocessing.md for original sources,
terms, exact exclusions and conversion details.

## Models and serving

| Label | Model | Immutable revision |
|---|---|---|
| PPLX | perplexity-ai/pplx-decider-v1-27b | d8c3bd18e31dc1804433d38528b3e770019bf00c |
| CLEF-flash | Cloudflare/clef-flash | 8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 |
| CLEF | Cloudflare/clef | 0b331204bb13fbd2ca93a64df1956f5b55478ce5 |
| Jev | TypeSafe official API | jev-1.13.0; reject mismatched response model |

Local models use their released Python loaders and heads, BF16, batch size 1 on
one Modal A100-80GB. PPLX uses its saved temperature; CLEF uses its released
softmax. A common 8192-token local input ceiling is enforced without truncation.
CLEF's official encoder normally truncates: we encode unbounded first and reject
overlong inputs instead. Jev uses its hosted context limits; local overlength
coverage differences must be reported. Gold labels and evidence annotations are
never sent to a model. No chain of thought, generation, ensembles or retries with
altered prompts. Model loader code is downloaded only at the pinned revisions.

API concurrency is 8. Transient requests receive at most 3 attempts with backoff;
auth failure stops the run. Credentials are read from .env and never saved in
configs or traces. One small Jev connectivity smoke test was successful before
the panel run; it is excluded from benchmark metrics.

Each GPU function has a hard 3600-second timeout; the runner stops at 3500 seconds
including model download/loading, reserving finalization time. User updated scheduling: run PPLX and CLEF-flash in parallel on two A100s.
CLEF requires review of their GPU cost first, and remains skipped if PPLX hits its cap.
This is the user's authorized one-GPU-hour-per-run policy. GPU setup and loader
incompatibilities count against the cap. Modal profile ac-worktrial was authenticated to workspace ac-worktrials.
The ac-worktrial environment was created because it did not exist. Periodic results-volume commits preserve partial traces.

## Metrics and evidence

Save append-only predictions including ID, label, group, full probability vector,
response, request hash (API), latency and errors. Probability vectors must be
finite, nonnegative and normalized within 0.025 rounding tolerance; retain their
original sum and normalize once. Model confidence in metrics is maximum option
probability; do not equate it to a vendor-specific confidence field.

Report accuracy, NLL, multiclass sum Brier, 15 equal-width-bin ECE, valid/expected
coverage, errors and median latency per benchmark. NLL uses a 1e-12 numerical
floor. Infrastructure failures remain errors, not ordinary incorrect predictions;
coverage-adjusted accuracy is separately saved. Missing models show Pending,
never fabricated zeroes. Report both macro accuracies once every panel ID is attempted: exclude misses from each suite denominator, or count misses as wrong. A running model remains Pending.
API latency includes network and retries; local latency is synchronized and
includes encoding. It is not a matched serving latency comparison. No uncertainty
intervals are claimed for this first descriptive pass.

## Reproduce

```bash
uv pip install --python .venv/bin/python -r requirements-preprocess.txt -r requirements-baselines.txt
.venv/bin/python scripts/preprocess_benchmarks.py
.venv/bin/python scripts/run_baselines.py freeze
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_baselines.py jev
.venv/bin/modal run --profile ac-worktrial --env ac-worktrial scripts/modal_baselines.py
.venv/bin/python scripts/export_baselines.py --models pplx clef-flash
# After cost-approved CLEF finishes:
.venv/bin/python scripts/export_baselines.py --models clef
.venv/bin/python scripts/run_baselines.py report
```

The profile/environment names follow the user request and require verification
against the authenticated account. GPU dependencies are pinned in modal_baselines.py;
host dependencies are in requirements-baselines.txt. Persisted model weights are
in `ac-worktrial-models`, traces in `ac-worktrial-baselines`.

Results: outputs/baselines-20261014/results.md and metrics.json, plus each model's
config.json, predictions.jsonl and completion.json. Never mix results from a changed
panel or runner into an existing directory. Jev can resume unchanged runs; GPU
reruns should use a fresh results directory to retain the original cap accounting.

## Launch history

Initial Modal app ap-0V52Fp53q5nhncCmOfWpLd was aborted after a container-import
error (Git invoked inside container). No inference results were produced. Fixed
by resolving the Git revision only locally and propagating it through the image.
User subsequently authorized two concurrent A100s for PPLX/CLEF-flash, with CLEF
cost-gated. One-hour hard timeout per function remains in place.

The export runner downloads each artifact to an explicit file path, verifies
existing evidence is byte-identical, and refuses differing overwrites.
Reference attention/convolution kernels were used in the first GPU runs because
flash-linear-attention and causal_conv1d were not installed. This affects runtime;
no optimized-serving claim is made. CLEF was approved after the cost review,
subject to the same one-hour cap and the PPLX-cap skip condition.

Reporting update: show one accuracy when coverage is 100%; otherwise show
valid-only / missed-as-wrong, with valid/expected counts. Completed runs with rejections still have both macro scores;
Pending means the model has not completed. Disagreement with gold is an error in
both variants. The JSON also retains valid-only disagreement rate and total
error rate counting missing predictions as wrong.
