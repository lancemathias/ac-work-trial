# Frozen-Qwen decision model work trial

This repository contains the accepted completed inference recipe and an auditable
record of head-training research on frozen Qwen backbones. No backbone fine-tuning
or LoRA was used. Measurements use a fixed 2,111-decision panel across eleven suites,
not a verified reproduction of vendor evaluation subsets.

The strongest completed inference recipe in the handoff ledger is **native plus
reversed-option readout: 79.49%** macro accuracy. The strongest completed trained
head is **E5: 68.41%**. They are different deliverables; the native recipe trains
no head. The matched Jev reference is 85.82%. Pending experiments are not promoted.

## Read the handoff

- [Chronological research report](docs/research-report.md): background, hypotheses, experiments, negative results, interpretation, pins and codebase map.
- [Presentation document](outputs/handoff/presentation.html) and [presenter notes](docs/presentation-notes.md).
- [Reproduction and inference instructions](docs/reproduce-handoff.md).
- [Accepted inference manifest](decision_model/accepted-model.json).
- [Per-run accepted checkpoint index](outputs/handoff/run-index.json).
- [All completed benchmark conditions](docs/performance-ledger.md).
- [W&B experiment workspace](https://wandb.ai/lmathias/ac-worktrial-decisions), including explicitly labelled [historical imports](outputs/handoff/wandb-imports.json).

## Inference

On the pinned Linux CUDA runtime, use `from decision_model import FrozenDecider`,
then `model = FrozenDecider.from_pretrained()` and
`model.predict(state, question)`. Or run `python -m decision_model request.json`.
The accepted recipe uses the original pinned Qwen3.8-27B backbone, two option
orders, semantic probability pooling and a saved calibration temperature. It
supports text-only Noul, Choice and ordered Score with 2–26 options and an
8,192-token ceiling. It does not start or deploy a Modal job automatically.
See the reproduction document for exact dependencies, numerical-runtime limits,
request format and the separately accepted trained-head artifact.

## Repo map

`decision_model/` is the focused inference path. `configs/`, `data/manifests/`,
`outputs/training/` and `outputs/test/` preserve reproducibility records. `scripts/`
contains historical preparation, evaluation and reporting tools; experiment-specific
runners live on their documented branches. `tests/` includes CPU contract and
artifact checks. `work/` is ignored scratch and independent worktrees.

The chronological report maps major commits and branches. `autoresearch/*` is
explicitly autoresearch; `preservation/20261015-*` contains archival snapshots.
The handoff branch consolidates research evidence. Main is updated through a PR
for an accepted candidate, without rewriting experiment history.

## Evidence and storage

W&B is the sole experiment tracker. Git stores source, reasoning, configurations,
seeds, hashes and compact evaluation evidence. Modal stores datasets, accepted
checkpoints and other saved states. Each run references only its accepted checkpoint;
other saved checkpoints remain reachable in the same indexed run directory.
Verified local tensor exports remain outside Git and outside Modal as a second copy.

Always use workspace `ac-worktrials`, profile `ac-worktrial`, environment
`worktrial-lance`. Do not create or use a Modal environment named `ac-worktrial`.
Do not overwrite finished runs, delete original checkpoints, or disturb active jobs.
Credentials, raw/processed dataset text and large model weights remain outside Git.
