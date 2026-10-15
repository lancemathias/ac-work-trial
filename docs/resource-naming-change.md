# Descriptive resource names — change inventory

**Applied naming change.** Current names, legacy aliases and verification receipts are described in [resource names](resource-names.md). The inventory below records the originally proposed scope. Replace the displayed experiment-number attribution with **Bidirectional full attention** and slug `attention-bidir`. Keep the `autoresearch` tag. The mixture becomes **Joint-head + native mixture**.

This is a naming/provenance cleanup: no inference or training rerun. Original identifiers and immutable evidence stay reachable. Historical file names containing E5 remain as legacy identifiers; current documents and aliases use descriptive names.

## W&B resources that would change

Project: `lmathias/ac-worktrial-decisions`. Run IDs and URLs stay fixed; display names change:

| Run ID | Current display name | Proposed display name |
| --- | --- | --- |
| [dyjus5wh](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/dyjus5wh) | `e5-fast-accuracy-bidir-v3-seed20261014` | `attention-bidir-qwen38-27b-v3-seed20261014` |
| [6spap1d0](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/6spap1d0) | `cached-e5-readout-equal-mixture-v1` | `cached-joint-head-readout-equal-mixture-v1` |
| [hist4d6cf18546e78e893ffd](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hist4d6cf18546e78e893ffd) | `Evaluation / E5 bidirectional` | `Evaluation / Bidirectional full attention` |
| [histbb478856e84c09dc39d4](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/histbb478856e84c09dc39d4) | `Evaluation / E5 + native mixture` | `Evaluation / Joint-head + native mixture` |

Artifact collection: create descriptive `attention-bidir-qwen38-27b-v3-seed20261014` from the original `e5-fast-accuracy-bidir-v3-seed20261014:v0` evidence, retaining the old collection/version. No tensor changes.

Existing run-history artifact IDs/names remain fixed. Retrospective evidence collections `hist4d6cf18546e78e893ffd-evidence` and `histbb478856e84c09dc39d4-evidence` remain immutable provenance; references to their run display labels are updated in current catalogs. The consolidated `research-handoff-20261015` artifact receives a new version after documentation regeneration.

## Modal resources

Workspace `ac-worktrials`; profile `ac-worktrial`; environment `worktrial-lance`. **No current/recent E5-named app was returned by Modal.** The source launcher would use `ac-worktrial-autoresearch-attention-bidir` for future launches instead of `ac-worktrial-autoresearch-e5`; this proposal does not launch it.

Create `resource-aliases/attention-bidir.json` in `ac-worktrial-training-results`. It gives these existing directories descriptive aliases, without moving/deleting their contents or breaking old checkpoint links:

| Volume | Existing path retained | Proposed alias |
| --- | --- | --- |
| `ac-worktrial-training-data` | `hidden-cache/e5-qwen38-27b-v3-bidirectional-fast-v1` | `hidden-cache/attention-bidir-qwen38-27b-v3-bidirectional-fast-v1` |
| `ac-worktrial-training-results` | `e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline` | `attention-bidir-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline` |
| `ac-worktrial-training-results` | `e5-accuracy-bidir-v3-seed20261014-pipeline` | `attention-bidir-accuracy-bidir-v3-seed20261014-pipeline` |
| `ac-worktrial-training-results` | `e5-fast-accuracy-bidir-v3-seed20261014-splits` | `attention-bidir-fast-accuracy-bidir-v3-seed20261014-splits` |
| `ac-worktrial-training-results` | `e5-fast-accuracy-bidir-v3-seed20261014` | `attention-bidir-fast-accuracy-bidir-v3-seed20261014` |
| `ac-worktrial-training-results` | `e5-fast-accuracy-bidir-v3-seed20261014-test` | `attention-bidir-fast-accuracy-bidir-v3-seed20261014-test` |
| `ac-worktrial-training-results` | `e5-fast-accuracy-bidir-v3-seed20261014-pipeline` | `attention-bidir-fast-accuracy-bidir-v3-seed20261014-pipeline` |
| `ac-worktrial-training-results` | `cached-e5-readout-equal-mixture-v1-calibration` | `cached-joint-head-readout-equal-mixture-v1-calibration` |

No volume, secret, model download or active app changes. Content-addressed handoff archive objects retain their original names/hashes; new current catalogs can be archived separately.

## GitHub resources

Repository: `lancemathias/ac-work-trial` over SSH. Apply changes on `deliverables/research-handoff` and its focused delivery copy `deliverables/accepted-model`, then push those branch tips. Main changes only through the user-created PR.

Add branch alias `autoresearch/attention-bidirectional` at `d9cefcbe1b2043667da3b6ecaaa291ccaabdcfb6`; retain `autoresearch/e5-attention-mask` and existing history.

### File renames

| Current path | Proposed path |
| --- | --- |
| `configs/e5-qwen38-27b-v3.json` | `configs/attention-bidir-qwen38-27b-v3.json` |
| `configs/e5-cache-source.json` | `configs/attention-bidir-cache-source.json` |
| `scripts/modal_e5.py` | `scripts/modal_attention_bidir.py` |
| `docs/autoresearch-e5-results.md` | `docs/autoresearch-attention-bidir-results.md` |

### Every current source/document/catalog file with an E5 name or reference

The files below would have their current labels, imports, path references or chart inputs updated. Historical receipt copies are listed separately and remain unchanged. Generated chart/presentation files below are regenerated from the renamed inputs.

- `docs/autoresearch-cutoff-audit.md`
- `outputs/handoff/autoresearch-cutoff-audit.json`
- `outputs/handoff/autoresearch-progress.json`
- `outputs/handoff/autoresearch-tags.json`
- `scripts/build_autoresearch_progress.py`
- `README.md`
- `configs/e5-cache-source.json`
- `configs/e5-qwen38-27b-v3.json`
- `docs/accepted-checkpoints.md`
- `docs/autoresearch-e5-results.md`
- `docs/autoresearch-results.md`
- `docs/autoresearch.md`
- `docs/native-readout-design.md`
- `docs/overnight-benchmark-scores.md`
- `docs/overnight-research-log.md`
- `docs/performance-ledger.md`
- `docs/presentation-notes.md`
- `docs/proof-of-concept.md`
- `docs/reproduce-handoff.md`
- `docs/research-log.md`
- `docs/research-report.md`
- `outputs/autoresearch/attention-repeat-preparation.json`
- `outputs/autoresearch/cached-e5-readout-mixture-calibration.json`
- `outputs/autoresearch/cached-e5-readout-mixture-dev.json`
- `outputs/autoresearch/cached-mixture-result-receipt.json`
- `outputs/autoresearch/cached-mixture-vs-readout-paired.json`
- `outputs/autoresearch/continuation.json`
- `outputs/autoresearch/e4-fast-native-cache-manifest.json`
- `outputs/autoresearch/e5-accuracy-cpu-validation.json`
- `outputs/autoresearch/e5-accuracy-launch.json`
- `outputs/autoresearch/e5-cache-identity-comparison.json`
- `outputs/autoresearch/e5-class-recall.json`
- `outputs/autoresearch/e5-control-backend-bidirectional.json`
- `outputs/autoresearch/e5-control-backend-differences.json`
- `outputs/autoresearch/e5-control-backend-native.json`
- `outputs/autoresearch/e5-dev-confusion.json`
- `outputs/autoresearch/e5-fast-attention-probe.json`
- `outputs/autoresearch/e5-fast-bidir-cache-manifest.json`
- `outputs/autoresearch/e5-fast-launch.json`
- `outputs/autoresearch/e5-launch.json`
- `outputs/autoresearch/e5-native-paired.json`
- `outputs/autoresearch/e5-split-export-receipt.json`
- `outputs/autoresearch/e5-training-snapshot.json`
- `outputs/autoresearch/e6-candidate-release.json`
- `outputs/autoresearch/events.jsonl`
- `outputs/autoresearch/experiment-cards.json`
- `outputs/autoresearch/overnight-benchmark-scores.csv`
- `outputs/autoresearch/overnight-benchmark-scores.json`
- `outputs/autoresearch/overnight-research-log.json`
- `outputs/autoresearch/performance-ledger.csv`
- `outputs/autoresearch/readout-export-receipt.json`
- `outputs/autoresearch/readout-vs-e5-paired.json`
- `outputs/autoresearch/source-addback-launch.json`
- `outputs/autoresearch/source-addback-live-status.json`
- `outputs/autoresearch/warm-handoff-stopped-apps.json`
- `outputs/autoresearch/warm-worker-driver-launch.json`
- `outputs/autoresearch/warm-worker-driver-v2-launch.json`
- `outputs/handoff/accepted-model-pr.md`
- `outputs/handoff/figures/completed-recipes.svg`
- `outputs/handoff/figures/task-transfer.svg`
- `outputs/handoff/presentation-notes.md`
- `outputs/handoff/reproduce-handoff.md`
- `outputs/handoff/research-report.md`
- `outputs/handoff/run-index.json`
- `outputs/handoff/wandb-imports.json`
- `scripts/apply_cached_mixture_test.py`
- `scripts/build_handoff_figures.py`
- `scripts/build_handoff_index.py`
- `scripts/build_handoff_presentation.py`
- `scripts/modal_attention_repeats.py`
- `scripts/modal_cached_mixture_calibration.py`
- `scripts/modal_e5.py`
- `scripts/modal_source_addbacks.py`
- `scripts/modal_warm_pipeline.py`
- `scripts/prepare_binary_views.py`
- `scripts/report_overnight_benchmarks.py`
- `scripts/research_snapshot.py`

### Generated resources to refresh

- `outputs/handoff/presentation.html`
- `outputs/handoff/figures/completed-recipes.png` and `.svg`
- `outputs/handoff/figures/task-transfer.png` and `.svg`
- `outputs/handoff/figures/autoresearch-progress.png`
- `outputs/handoff/research-report.md` and `presentation-notes.md`
- New `configs/resource-aliases.json` and matching Modal alias catalog.

### Historical evidence retained under original names

These paths contain E5 identifiers but would not be edited or renamed. Their hashes and upstream resource references remain valid; the alias catalog provides the descriptive current name.

- `outputs/autoresearch/snapshots/20261015T045428Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/cpu-validation.json`
- `outputs/autoresearch/snapshots/20261015T045428Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/status.json`
- `outputs/autoresearch/snapshots/20261015T045612Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/cpu-validation.json`
- `outputs/autoresearch/snapshots/20261015T045612Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/status.json`
- `outputs/autoresearch/snapshots/20261015T045805Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/cpu-validation.json`
- `outputs/autoresearch/snapshots/20261015T045805Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/status.json`
- `outputs/autoresearch/snapshots/20261015T050450Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/cpu-validation.json`
- `outputs/autoresearch/snapshots/20261015T050450Z/e5-qwen38-27b-bidir-v3-seed20261014-af01134-pipeline/status.json`
- `outputs/handoff/export-metadata/cached-e5-readout-equal-mixture-v1/checksums.json`
- `outputs/handoff/export-metadata/cached-e5-readout-equal-mixture-v1/completion.json`
- `outputs/handoff/export-metadata/cached-e5-readout-equal-mixture-v1/config.json`
- `outputs/handoff/export-metadata/cached-e5-readout-equal-mixture-v1/metrics.json`
- `outputs/handoff/export-metadata/cached-e5-readout-equal-mixture-v1/temperature.json`
- `outputs/handoff/export-metadata/e4-fast-accuracy-v3-seed20261014/config.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014-splits/checksums.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014-splits/manifest.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014-test/checksums.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014-test/completion.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014-test/config.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014-test/metrics.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014/best.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014/checksums.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014/config.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014/summary.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014/temperature.json`
- `outputs/handoff/export-metadata/e5-fast-accuracy-bidir-v3-seed20261014/wandb-run.json`
- `outputs/handoff/final-artifacts.json`
- `outputs/handoff/modal-archive.json`
- `outputs/test/cached-e5-readout-equal-mixture-v1/checksums.json`
- `outputs/test/cached-e5-readout-equal-mixture-v1/completion.json`
- `outputs/test/cached-e5-readout-equal-mixture-v1/config.json`
- `outputs/test/cached-e5-readout-equal-mixture-v1/export-receipt.json`
- `outputs/test/cached-e5-readout-equal-mixture-v1/metrics.json`
- `outputs/test/cached-e5-readout-equal-mixture-v1/prediction-distributions.json`
- `outputs/test/cached-e5-readout-equal-mixture-v1/results.md`
- `outputs/test/cached-e5-readout-equal-mixture-v1/temperature.json`
- `outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/checksums.json`
- `outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/completion.json`
- `outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/config.json`
- `outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/export-receipt.json`
- `outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/metrics.json`
- `outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/pipeline.json`
- `outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/results.md`
- `outputs/test/readout-fast-qwen38-27b-v1/export-receipt.json`
- `outputs/traces/cached-e5-readout-equal-mixture-v1/calibrated-predictions.jsonl.gz`
- `outputs/traces/cached-e5-readout-equal-mixture-v1/raw-predictions.jsonl.gz`
- `outputs/traces/e5-fast-accuracy-bidir-v3-seed20261014-test/calibrated-predictions.jsonl.gz`
- `outputs/traces/e5-fast-accuracy-bidir-v3-seed20261014-test/raw-predictions.jsonl.gz`
- `outputs/traces/e5-fast-accuracy-bidir-v3-seed20261014/calibration-traces.jsonl.gz`
- `outputs/traces/e5-fast-accuracy-bidir-v3-seed20261014/dev-traces.jsonl.gz`
- `outputs/traces/e5-fast-accuracy-bidir-v3-seed20261014/metrics.jsonl.gz`
- `outputs/traces/index.json`
- `outputs/training/e4-fast-accuracy-v3-seed20261014/config.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/best.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/checksums.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/config.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/dev-comparison.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/export-receipt.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/protocol-match.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/summary.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/temperature.json`
- `outputs/training/e5-fast-accuracy-bidir-v3-seed20261014/wandb-run.json`

A machine-readable copy is [resource-naming-impact.json](../outputs/handoff/resource-naming-impact.json). The inventory excludes unrelated hexadecimal strings containing the characters `e5`. Large/ignored source exports are retained as historical artifacts, not rewritten.
