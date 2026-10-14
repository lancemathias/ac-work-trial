# Frozen hidden-state cache for W&B run nulun3tv

This cache uses the live configuration read from [W&B run nulun3tv](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/nulun3tv). The run snapshot is frozen in `configs/hidden-cache-nulun3tv.json`. Extraction reads all 5,000 points from the verified typed dataset, including its existing train/dev/calibration families; it does not change the split assignment or optimize backbone weights.

## Exact feature contract

- Backbone: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Tokenizer/processor: the same repository and immutable revision.
- Renderer: the live training adapter and unmodified Clef encoder, copied under `scripts/hidden_cache_runtime/`. Their SHA-256 hashes are checked against the W&B configuration before extraction.
- Features: **every token's final normalized `last_hidden_state`**, in native BF16. These are not pooled vectors, selected-token summaries or all-layer outputs. The existing joint head attends to the full sequence and also uses question/option spans.
- Forward: same text-model submodule as training; frozen weights, `eval()`, `torch.no_grad()`, `use_cache=False`, SDPA, TF32 disabled and cuDNN SDPA disabled. No generation or sampling temperature applies.
- Shape/batching: one unpadded point per forward, matching the live training loop. The 8,192-token limit rejects overlength inputs; no silent truncation is introduced.
- Seed: `20261014`, copied from the W&B configuration for CPU and CUDA RNGs. Fresh-head parity checks use the same seed but never update that head's weights.
- Runtime: pinned training dependencies from `requirements-hidden-cache.txt`; the extractor checks PyTorch and Transformers versions against the live run.

## Modal location

Workspace `ac-worktrials`, profile `ac-worktrial`, environment `worktrial-lance`.

```text
Volume: ac-worktrial-training-data
Directory: /hidden-cache/nulun3tv-qwen35-9b-3b3d8db82da7
```

The extractor shares the existing A100-80GB training container in a separate process; it does not stop, modify or resume the running training job. The training loop must adopt the cache reader to avoid future backbone forwards.

## Files and consumption

`manifest.json` records completion, extraction identity, source counts and file hashes. `index.json` maps each original dataset ID to a shard, token offset/count, encoded question/option spans, source, group and native training targets. `shards/NNNN.safetensors` contain concatenated `hidden_states` of shape `[sum_token_lengths, hidden_size]` and `input_ids` of shape `[sum_token_lengths]`. Shards contain at most 64 points; there is no padding in stored features. All attention-mask entries for an individual restored point are one.

`output-embedding-rows.safetensors` preserves the frozen LM output-embedding rows for every token appearing in option spans. The existing head uses those rows for lexical features and its prior. The cache includes a lookup object implementing the head's `weight[token_ids]` interface, so the full 9B backbone is not needed merely to retrieve those embeddings. Unknown lexical IDs fail rather than receiving invented embeddings.

`source-run.json` preserves the W&B configuration snapshot; `extraction-config.json` records the actual feature identity. `progress.json` supports checked shard-level resumption. A different identity cannot overwrite an existing partial cache. Every saved shard has a SHA-256 checksum.

Example use with the existing `JointSchemaHead`:

```python
import sys
sys.path.insert(0, "/project/scripts")
from cache_hidden_runtime import HiddenCache

cache = HiddenCache("/data/hidden-cache/nulun3tv-qwen35-9b-3b3d8db82da7", device="cuda")
# Initialize/load the existing head, on CUDA, using its recorded architecture.
# Retain the current split membership, targets, field loss weighting and optimizer.
logits = cache.forward(head, record_id, autocast=True)[0]
# Compute the usual supervised loss, then backward/optimizer.step on head only.
```

`cache.get(record_id)` exposes the restored tensors and encoded record if a custom loop needs them. `cache.forward` returns the same nested per-record/per-question logits structure as the existing head. It leaves gradient tracking enabled for head training. Cached hidden states and lexical embeddings remain frozen.

## Verification and durability

Extraction checks the dataset and training-point-schema hashes, all source IDs, maximum token length, runtime versions and finite hidden states. Three direct-versus-cached head probes verify exact logits with the native lexical embedding lookup and cached sparse lookup. The final cache reader rechecks all files and performs head backward on three cached points. These are compatibility tests, not model-quality measurements.

CUDA reproducibility across different hardware, drivers or kernels is not guaranteed. The preserved cache bytes and their checksums define this artifact exactly; using those bytes avoids repeat backbone computation and its numerical variation.

Large tensors are exported locally with checksums before any Modal cleanup. The local export and remote completion receipt are recorded under `outputs/hidden-cache/`. No cache or source code should be deleted merely to clean temporary Modal resources.

Replay extraction:

```sh
/Users/bwt/ac-worktrial/.venv/bin/modal run --profile ac-worktrial --env worktrial-lance scripts/modal_hidden_cache.py
```

This uses the frozen configuration snapshot, not a mutable query for whichever W&B run happens to be newest. The separate `--inspect-only` mode is for read-only run discovery.

## Attention and complete backbone metadata

The `backbone-metadata/` directory preserves the model and tokenizer configurations, special tokens, parameter names/shapes/dtypes, source-weight SHA-256 hashes, installed package versions, CUDA/cuDNN/driver settings, PyTorch binary hashes, and the loaded Transformers implementation files. Missing ancillary HF repository files are downloaded only at the pinned revision; model weights are reused from the existing cache.

Each index entry records the actual input mask shape/dtype/hash, causal and recurrent mask factory outputs, and observed SDPA calls including explicit-mask presence, `is_causal`, dropout, scaling, GQA and query/key/value shapes. An absent explicit mask does not imply bidirectional attention: SDPA can enforce causality with `is_causal=True`. A future-suffix perturbation probe verifies that earlier token states are unchanged for a test input. The pinned recurrent implementation and hybrid layer types are preserved alongside these observations.

## Separate Qwen3.8-27B worker

The requested second worker uses `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The source W&B configuration is explicitly overridden only for the backbone ID/revision; it is a derived configuration, not a claim that the 9B W&B run used the 27B model. See `configs/hidden-cache-qwen38-27b.json` and `outputs/hidden-cache-qwen38-27b/launch.json`. The same renderer, dataset, no-truncation policy, BF16/SDPA settings and seed are retained; the new model's actual dimension, tokenizer, layer types, masks and runtime are independently captured.

Remote directory: `/hidden-cache/qwen38-27b-1d4bf0f2ff60-3b3d8db82da7`. Local export: `data/hidden-cache/qwen38-27b`. Launch with `modal run --profile ac-worktrial --env worktrial-lance scripts/modal_hidden_cache27.py`. The caches are separate artifacts and must not be mixed or passed to a head expecting the other backbone dimension.
