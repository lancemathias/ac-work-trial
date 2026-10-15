# Reproduce the handoff

The accepted completed inference recipe is in `decision_model/`; research history
is preserved on `autoresearch/*` and `preservation/20261015-*`. No command below
starts a Modal GPU job automatically. Use Python 3.12 for the measured runtime.

## Restore and verify measured results without inference

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-preprocess.txt
.venv/bin/python scripts/preprocess_benchmarks.py
.venv/bin/python scripts/restore_handoff_traces.py
.venv/bin/python scripts/report_overnight_benchmarks.py
```

The restore command verifies compressed and decompressed hashes and refuses
conflicting existing trace bytes. It restores compact committed export metadata,
including manifests previously available only in ignored local exports. The report
checks source hashes, panel identity, scored predictions and all composite means.
It uses no model or API. Input downloads still require network access. The report
regenerates timestamps; compare numerical scores and source hashes, not timestamp
bytes. Regenerate the four figures with `scripts/build_handoff_figures.py` in an
environment with matplotlib; archived PNG and SVG files are included.

## Accepted inference recipe

On a Linux CUDA machine with sufficient memory (measured on A100-80GB):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-training.txt
.venv/bin/pip install --no-deps -r requirements-kernels.txt
.venv/bin/python -m decision_model request.json --cache-dir /models/hub
```

The measured CUDA runtime is Torch 2.9.1+cu128 with the pinned optimized kernels.
Use the matching CUDA wheel build, as recorded in the run configuration; the
causal-conv1d wheel is explicitly Linux CPython 3.12 / Torch 2.9 / CUDA 12.
The inference loader rejects a different kernel identity. It loads only the
original pinned Qwen backbone, keeps all weights frozen, makes native and reverse
option-order passes, aligns semantic IDs, averages uncalibrated probabilities and
applies the accepted pooled temperature. No optimizer, LoRA, temperature fitting
or benchmark labels are involved.

Example `request.json`:

```json
{
  "state": "The requested refund was issued yesterday.",
  "question": {
    "type": "choice",
    "instructions": "Which team should receive this request?",
    "criteria": {
      "billing": "Payments and refunds",
      "technical": "Software integration problems"
    }
  }
}
```

For Python, use `from decision_model import FrozenDecider`, then
`model = FrozenDecider.from_pretrained()` and `model.predict(state, question)`.
Noul, Choice and ordered Score share the scorer. The accepted recipe supports
2–26 options and rejects inputs above 8,192 tokens; it is not a claim to support
all 255 options of the experimental residual head. Confidence for Choice/Score
is the maximum class probability. Multiple questions should be called separately;
this delivery path has no joint-field coupling or image input support.

The packaged prompt, selected-logit operation and semantic pool preserve the
original research implementation. CPU contract tests verify prompt agreement
and reproduce all valid archived pooled distributions to 12 decimal places.
The measured GPU run remains the performance evidence: packaging did not launch
a new full-model GPU validation. Per-request inference uses batch size one;
reported benchmark latency used batches of four. Do not treat that latency as
a per-request service guarantee or promise cross-batch bitwise equality.

## Accepted trained head and other checkpoints

`outputs/handoff/run-index.json` records each run's accepted head, exact SHA-256,
selection epoch/step, W&B link and Modal location. The strongest completed trained
head in the indexed panel is bidirectional full attention; the accepted native/reverse recipe is a different,
untrained-head inference approach. Its protocol digest is not a checkpoint tensor
hash. The separate checkpoint index preserves that distinction.

Download a specific accepted checkpoint using its exact indexed path:

```bash
modal volume get --profile ac-worktrial --env worktrial-lance \
  ac-worktrial-training-results '<accepted path from run-index.json>' \
  work/accepted-head.safetensors
```

Verify SHA-256 before loading. Other saved checkpoints remain reachable in the
same indexed run directory. Do not select a different epoch to match a benchmark.
To reproduce training use the original run config and code commit, not the latest
working tree or a newly sampled dataset. Use the original feature-cache identity
and its replay gate, or re-extract the frozen features under the exact runtime.
Training and Modal entrypoints remain documented on their experiment branches;
`docs/first-experiment-training.md`, `docs/autoresearch.md` and the run index map
those historical protocols. Earlier NLL selection remains historical; later
accuracy selection does not retroactively change the accepted checkpoint.

## Verification and artifact storage

Run CPU checks with the pinned training dependencies:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

W&B is the authoritative tracker. `outputs/handoff/wandb-imports.json` links the
retrospective scored-condition imports. Modal environment `worktrial-lance`
contains datasets, accepted checkpoints and other saved states. Data manifests
remain in Git; dataset text and tensor exports remain outside it. No script in
the handoff cleanup stops active jobs or removes original resources.
