# Modal work-trial environment

For the Applied Compute work trial in `/Users/bwt/ac-worktrial`, use Modal
workspace `ac-worktrials`, profile `ac-worktrial`, and environment
`worktrial-lance` for every future run, deployment, volume, and secret lookup.
The profile name is not the environment name. Do not create or use a Modal
environment named `ac-worktrial`. This is the user's instruction of 2026-10-14
and applies across sessions. Keep research code and durable results outside
Modal: version control code, configs, seeds, traces, metrics and documents;
export large tensor artifacts locally with checksums before cleaning Modal.
Do not revert research code or results when cleaning up Modal resources.

# Frozen readout display names

Use exactly these display names for frozen pretrained readout baselines in prose,
tables and figures: `Qwen3.5-9B readout`, `Qwen3.8-27B readout`,
`Qwen3.5-9B + reverse`, `Qwen3.8-27B + reverse`. Record prompt/codebook variants
separately from the display name. Preserve immutable run IDs, paths, raw metric
keys and historical receipts.

# Experiment branches and promotion

Keep one dedicated Git branch per numbered experiment. Use the branch map in
`configs/experiment-branches.json`; branches may reference prior experiments.
Put future Experiment 10 work on `codex/experiment-10-v4-residual-l2-sweep`,
not the shared research-handoff branch. The user will create and merge the final
Experiment 10 PR to `main`; do not create that PR or merge it on their behalf.
