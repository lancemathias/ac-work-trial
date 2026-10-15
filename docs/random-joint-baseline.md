# Frozen Qwen with an untrained joint schema head

This baseline evaluates the original Qwen3.8-27B checkpoint with three independently
initialized instances of Cloudflare's Clef JointSchemaHead. No gradient updates,
LoRA, adapted Clef backbone weights, released head weights, temperature fitting
or benchmark-driven seed selection are used.

Full layer, tensor-shape, scoring and precision details are in the
[joint head architecture](joint-head-architecture.md).

## Frozen protocol

The evaluation uses the existing 2,111-example matched baseline panel across all
11 research-log tasks, not the full 65,377-example processed collection. Each seed
sees the same 200 examples per suite and all 111 JevBench public hard examples.
The panel SHA-256 and upstream data hashes are retained in each run configuration.
See [the baseline protocol](baselines.md) for selection and conversion details.

- Backbone: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Head code: `Cloudflare/clef` at
  `0b331204bb13fbd2ca93a64df1956f5b55478ce5`, source SHA-256
  `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
- Initialization seeds: **20261014, 20261015, 20261016**. Python, CPU Torch and
  CUDA RNGs are seeded before each CPU FP32 head construction, then the head is
  converted to BF16. All three exact initial states are saved as safetensors.
- Native head dimensions: hidden 5,120; width 1,024; two evidence-routing layers;
  four joint transformer layers; 16 heads; feed-forward width 4,096; dropout zero.
- Original backbone weights are frozen and run in evaluation/inference mode.
  Parameter version counters and gradient flags are checked before/after the run.
  No optimizer exists. Native causal backbone attention is retained.
- One A100-80GB, batch size one, BF16, SDPA; common 8,192-token ceiling without
  truncation. One serialized worker has a one-hour hard cap, with a 3,500-second
  soft budget including loading and all three heads. Failures remain explicit.

## What random initialization means here

The Clef constructor includes deterministic constants, randomly initialized
projections/transformers, and a lexical prior computed from the frozen backbone's
output embeddings. Retain that published architecture unchanged. This is an
**untrained head with a pretrained lexical prior**, not a fully random classifier
or a guarantee of chance accuracy. The three seeds vary head initialization only.
The official release loader is not used because it loads an adapted backbone and
trained head.

The original tokenizer, Clef schema encoding, question/option spans and frozen
output-embedding vectors feed the new head. Send only state/question content to
encoding, never labels or provenance. Preserve full-precision softmax probabilities
in the traces instead of the released response helper's four-decimal rounding.
No temperature adjustment is applied.

## Scoring and averaging

Compute each seed's accuracy separately, using argmax for Choice, Noul and Score.
For BBH, average its included subtask accuracies first. Then average the 11 suite
accuracies equally to produce each seed's macro score. Finally report the mean
and sample standard deviation across the three seed scores. Do not average
probabilities into an ensemble, pool the three copies as independent examples,
or select the best seed.

Results-table cells report mean accuracy ± sample standard deviation in percentage
points, followed by valid counts for seeds in saved order and the expected count.
This standard deviation measures initialization sensitivity, not a confidence
interval over benchmark examples. The strict complete-valid macro is undefined
if any example is missing/invalid for any seed. To match the results table's
current convention, affected cells and macro show valid-only accuracy followed
by coverage-adjusted accuracy (misses counted as wrong). Both are averaged across
seeds separately; the complete-valid field remains null in metrics.json.
Macro values are reported only when all IDs have been attempted.
Per-seed NLL, sum Brier and 15-bin ECE are stored alongside their mean and sample
standard deviation, with no calibration fitting.

For efficiency, compute each example's frozen hidden states once and immediately
apply all three heads. There is no persistent token-level feature cache. Traces
save shared backbone/encoding time and each head's time separately; their sum is
the component estimate for one pipeline, not an independently timed serving run.
Record GPU function wall time separately for actual compute accounting.

## Artifacts and reproduction

Launch `scripts/modal_random_joint.py` with the authenticated `ac-worktrial`
profile and `worktrial-lance` environment for future runs. Download its artifact directory from the
`ac-worktrial-baselines` Modal Volume at
`baselines-20261014/random-joint-qwen38-27b/`.

The directory includes `config.json`, `completion.json`, `metrics.json` after local
reporting, and one `seed-<seed>/` folder per initialization. Each seed folder
contains its config, initial head safetensors with SHA-256, and append-only
predictions containing probabilities, labels, group IDs and failures.

Run `scripts/run_baselines.py report` after export to regenerate the existing
results table with the new averaged column. Methodology stays in this document;
the results table contains measured values and coverage. Raw benchmark corpora
remain outside Git. This is our matched protocol, not a reproduction of the
unpublished PPLX panel.

Local configs, traces and the completion receipt can be exported with
`scripts/random_joint_baseline.py --export`. Initial head tensors were also exported to the local archive
`outputs/modal-archive-20261014/baselines-20261014/` before cleaning up the old
Modal environment. Their individual hashes are retained in the seed configs;
the routine trace export does not duplicate those roughly 512 MB files. The exact GPU
runner source is archived locally as `runtime-runner.py`, matching the runner
hash recorded by the job. Local score verification uses a 1e-12 float tolerance
for cross-platform/Python-version arithmetic, while checking predictions exactly.

## Execution receipt

Modal app: `ap-Tymk8pKGmiFWtNQ495V1Dq`, profile `ac-worktrial`, environment
`ac-worktrial`. Completed all 2,111 attempts per seed. One ContractNLI request per
seed exceeded 8,192 tokens and was explicitly rejected; every other request
returned valid probabilities. Backbone parameter versions and frozen-gradient
flags were unchanged. GPU worker elapsed time was 1,410.40 seconds including
download/loading and all heads; peak allocated memory was 58,545,682,432 bytes
(54.52 GiB). Reference PyTorch convolution and Gated DeltaNet kernels were used;
this is not an optimized-serving measurement. Transport logs timed out while
waiting for final app logs, but the GPU function returned successfully and its
persisted completion receipt and prediction coverage were independently verified.

`seeds.json` lists the three seeds and head-state hashes. `artifact-manifest.json`
records SHA-256 hashes of every locally exported evidence file.
