# First experiment: frozen Qwen + joint schema head

Implementation: [training loop](../scripts/train_joint.py),
[data guards](../scripts/training_data.py), [Modal launcher](../scripts/modal_train_joint.py),
[run config](../configs/e1-qwen35-9b.json), [exporter](../scripts/export_training.py).
The initial dataset is pinned in `worktrial-lance`, Volume
`ac-worktrial-training-data`, at
`/typed-5000-3b3d8db82da7/typed-5000.jsonl`. SHA256:
`3b3d8db82da7595565ad48c8cb7da981821097324c151169cf64c4c11fc3ea36`.
Its locally available identical copy passed schema/adapter validation: 5000
records, 5500 fields (3875 Choice, 1000 Noul, 625 Score). Split approval is pending;
no GPU training job has been launched.

E1 loads the original `Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, freezes every backbone parameter,
and initializes a fresh Clef `JointSchemaHead`. The pinned Apache-2.0 implementation
and license are [vendored](../scripts/vendor/README.md); released Clef weights
are not used. No LoRA/adapters, backbone updates, generation or RLCD in E1.

Head training uses cross-entropy over each question's valid alternatives and
AdamW: LR 1e-3, weight decay 0.01, 10% warmup followed by linear decay, gradient
clip 1, up to five epochs and two non-improving development epochs before stopping.
The optimizer receives head parameters only. Backbone extraction uses `no_grad`
and eval mode; the FP32 head runs under BF16 CUDA autocast. Microbatch size is one
request; gradients accumulate to at least 64 decisions and are normalized by the
actual decision count. Requests with multiple questions may cross that threshold.
Token sequences above 8192 are rejected, never silently truncated. Feature caching
remains a deferred TODO; E1 currently computes features on each pass.

Select the best head using raw development NLL. Freeze it, then fit **one positive
scalar temperature** on the independent calibration partition by minimizing NLL.
The bounded search uses T in [0.05, 20] and retains T=1 if fitting does not improve
calibration-set NLL. These bounds are a provisional experimental choice, recorded
with the fit. Calibration-set results are in-sample fit diagnostics; development
results are reported raw and calibrated. Accuracy uses modal class/level; ordinal
Score MAE uses the probability-weighted level. ECE uses 15 equal-width bins and
maximum class probability, not a derived API confidence field.
Development metrics also report source and primitive slices to expose regressions
hidden by aggregate decision counts.

RLCD-style head-only logit tuning is a follow-up. Its objective and algorithm must
be specified before implementation; it is not a claim to reproduce Jev's recipe.

## Supplied training-point contract

The loader uses the supplied [training-point.schema.json](../configs/training-point.schema.json),
copied unchanged from the user's `cacc` worktree. Contract SHA256:
`bdf1f5368d395ae173949c4b9bf95300c0e4db5b4350ed4dba3bcaa19c5e94ad`.
Each JSONL point contains `id`, `source`, `family_id`, `model_payload`, `gold`,
`field_types`, generator/native provenance and audit-only metadata.

Only `model_payload` is converted to the model's input: its `input` must be a
canonical JSON string with exactly `state` and `question`, and its flat output
schema defines boolean or string-enum fields. Gold, target metadata,
transformation traces and generator/native provenance never enter the encoder.
Unlike protected benchmark storage, this supplied training format keeps gold in
the same row; isolation is enforced by the adapter's explicit input whitelist.

The adapter validates both the outer contract and gold against the output schema,
then requires identical keys in properties, required fields, gold and field_types.
It constructs Jev-shaped internal requests: boolean Noul fields map to true/false;
Choice enums map to allowed option keys; Score enums retain their ordinal array
order and map gold strings to level indices. Score needs 2..10 levels. Numeric
expected scores remain continuous at evaluation time; hard observed level labels
train the distribution. No confidence labels or fractional model outputs are used
as gold. Human soft vote distributions are not supported in E1.

The question and field description become structured instructions. Choice option
descriptions already occur in the allowed field description; the adapter never
extracts semantic information from audit-only `transformation_trace`. The Clef
encoder sorts Choice keys; target indices come from its encoded IDs, so gold stays
aligned. It includes question IDs and jointly processes the schema, unlike Jev's
documented ID-independent/question-independent behavior. This is a compatible
input adapter, not a recreation of the Jev implementation. Media are out of scope.

## Approved splits

Use [prepare_training_splits.py](../scripts/prepare_training_splits.py) after the
upstream audit supplies a report with `approved: true` and the exact
`dataset_sha256`. The helper does not fabricate approval or run an LLM judge.
It joins families linked by identical states and deterministically assigns those
connected groups to train/dev/calibration with 80/10/10 SHA256 thresholds and seed
20261014. Actual counts vary; it rejects empty roles. Near-duplicate and benchmark
containment screening remain upstream requirements, not capabilities of this
splitter. Existing conversion/audit metadata must be reviewed against the allowed
data policy before approval.

The resulting `manifest.json` has version 1, source dataset/schema hashes, split
seed/rule, a checksummed approved hygiene report, and exactly three splits. Each
split has this shape:

```json
{
  "role": "train",
  "evaluation_only": false,
  "allowed_sources": ["ANLI", "ARC"],
  "count": 4000,
  "points": {"path": "train.jsonl", "sha256": "[actual SHA256]"}
}
```

The count above illustrates the format, not a measured split. All paths stay
inside the manifest directory. Loading rejects schema/hash changes, duplicate
JSON keys/IDs, inconsistent gold, unauthorized sources/roles, evaluation-only
splits and family/exact-state/request overlap across roles. The manifest must
match the source dataset pinned in the E1 config. The Modal worker mounts the
training Volume and no test Volume. A manifest cannot detect an incorrect audit
attestation or unknown backbone pretraining contamination.

## Tracking, execution and export

**W&B is the single experiment tracker.** The committed config uses online mode;
the loop records config/provenance, training loss/LR/gradient norm/throughput,
untrained and trained development metrics, fitted temperature and final results.
W&B evidence artifacts include config, metrics, calibration and development traces;
metadata references/checksums identify binary checkpoints retained in Modal.
`offline`/`disabled` modes exist for local diagnostics, not tracked E1 runs.
Modal provides resource diagnostics and execution logs. JSONL files, checkpoints
and checksummed exports are durable evidence copies, not additional trackers.

Before running: create grouped approved partitions and publish their manifest
directory alongside the pinned unsplit dataset in Volume
`ac-worktrial-training-data`; configure Secret `ac-worktrial-wandb` containing
only `WANDB_API_KEY` from the ignored primary checkout `.env`. Do not upload the
entire `.env` or embed credentials in code/images. The launcher uses one
A100-80GB, at most one container and a 12-hour timeout. No resource is provisioned
by adding this code.

```sh
modal run --profile ac-worktrial --env worktrial-lance scripts/modal_train_joint.py --manifest e1/manifest.json --run-id e1-qwen35-9b-seed20261014
python scripts/export_training.py --run-id e1-qwen35-9b-seed20261014 --destination outputs/training/e1-qwen35-9b-seed20261014
```

Workspace must be `ac-worktrials`. Volume/Secret lookups explicitly use
`worktrial-lance`; the launcher rejects other profiles/environments. Export the
complete run locally and verify every file checksum **before** any Modal cleanup.
The exporter does not delete Modal resources or overwrite local evidence. Version
control configs, seeds, metrics, traces and documents; retain large binary tensor
exports locally with their checksums rather than committing them to Git.

Artifacts: initial and best head safetensors; last head/optimizer/RNG state;
best-epoch selection; temperature; raw/calibrated JSONL traces; metrics; summary;
run provenance; W&B run link; checksums; failure record if an exception occurs.
Epoch checkpoints are for inspection; automatic interrupted-run resume is not
implemented. CUDA nondeterminism and seed variability still need measured repeats.

## Validation

Run `python -m unittest discover -s tests -p test_training.py -v` with
`requirements-training.txt` installed. CPU checks use the actual vendored joint
head and a tiny test backbone, covering label alignment, mixed question types,
head learning with identical frozen backbone tensors, temperature fitting,
split/hash/leakage guards, full entrypoint artifact output and truncation rejection. Tiny fixtures are test-only,
not synthetic training data for E1. Actual Qwen/A100 throughput, memory and BF16
behavior remain to be validated with the approved dataset.


## Runtime validation after initial launch failures

Resolve both data mount and manifest before containment checks: Modal mounts may
be symlinks. Pin `torchvision==0.24.1` alongside `torch==2.9.1`; Qwen's processor
loads media components even for text-only requests. Modal injects credentials
from the dedicated Secret; the image does not need a .env file.

The exporter reads each regular file in the checksum manifest through the Modal
SDK, verifies its digest, and publishes only a complete local copy. Recursive
CLI download failed with Modal 1.6.1. Legacy failed runs containing W&B symlinks
still cannot be fully exported because Modal refuses to read those links; their
remote evidence is retained. New runs keep W&B files outside the run directory.

Validation: 23 local tests passed, including actual Qwen hybrid BF16 forward and
joint-head backward with no backbone gradients. All 5,000 approved requests passed
the pinned processor/tokenizer: train 4,101 requests / 4,489 decisions, dev 458 /
514, calibration 441 / 497. Maximum lengths: 2,508 / 2,085 / 2,093 tokens. A clean
remote fixture passed upload, SDK download, checksum verification and publication.
Full-model CUDA behavior and throughput still require the real run. Optional
accelerated linear-attention kernels are absent; speed must be measured.
