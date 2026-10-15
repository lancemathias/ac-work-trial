# Deterministic typed training generator and training-point contract

The deterministic generator has processed the 5,000 screened native training records. Nimble was skipped following the user's instruction: all labels already exist, and explicit local rules suffice. No teacher model, API, GPU, Modal operation, model prediction or confidence estimate was used.

One native datapoint becomes one training point. There are 5,000 points, with 5,500 supervised fields: 3,875 Choice, 1,000 Noul and 625 Score fields. HelpSteer2's 125 rating records each contribute five fields; all other records contribute one. These are hard training targets, not probabilities.

## Explicit training-point schema

The complete machine-readable contract is [training-point.schema.json](../configs/training-point.schema.json), using JSON Schema draft 2020-12. A real, complete sentiment training point is [training-point-example.json](training-point-example.json). Every record has exactly these top-level fields:

| Field | Type | Meaning and training use |
|---|---|---|
| `id` | string | Original sampled record ID, globally unique in this corpus. |
| `source` | candidate-source enum | Source dataset, one of the frozen 20-source allowlist. |
| `family_id` | string | Source family/group ID. Use for partitioning before any further expansion. |
| `model_payload` | object | The only fields to pass to the model: `input` and `schema`. |
| `model_payload.input` | string | Canonical JSON text containing `state` and `question`. State can originally be a string, list or object; it remains inside this serialized context. |
| `model_payload.schema` | flat object schema | Required boolean or string-enum decision fields, with descriptions, no additional properties. Option meanings are explicitly included in descriptions. |
| `gold` | object mapping field names to boolean or string | Supervised targets. Its keys exactly equal the schema's properties and required fields. Every enum target must belong to its enum. |
| `field_types` | object mapping the same keys to `noul`, `choice`, or `score` | Semantic task type. Noul is boolean; Choice is string enum; Score is an ordered numeric-string enum. |
| `target_metadata` | object | Native annotation fractions or preference strength retained for audit only. Not an input or confidence field. |
| `transformation_trace` | array of objects | Explicit conversion operations, option permutations and inverse mappings, rubric provenance, thresholds and synthetic-answer provenance. Not model input. |
| `generator` | object | Generator version, seed and exact hash-ordering algorithm. |
| `native_source` | object | Native split/config/revisions/row indices or file provenance available in the original record. Full acquisition pins are in source manifests. |

The central payload/target pattern is:

```json
{
  "model_payload": {
    "input": "{\"question\":\"Which category applies?\",\"state\":\"Example text\"}",
    "schema": {
      "type": "object",
      "properties": {
        "decision": {
          "type": "string",
          "enum": ["option_0", "option_1"],
          "description": "Select exactly one allowed option. Options: option_0 means category A; option_1 means category B."
        }
      },
      "required": ["decision"],
      "additionalProperties": false
    }
  },
  "gold": {"decision": "option_1"},
  "field_types": {"decision": "choice"}
}
```

This excerpt is illustrative, not an additional training point; the linked JSON example includes the full envelope. A Noul decision uses `type: boolean` and a boolean gold value. A Score field uses the fixed enum `["0","1","2","3","4"]` with an explicit rubric and a matching string gold value. Decode Score strings as integer ordinal levels for a score head; do not treat arbitrary category IDs or math answers as ordinal scores. The outer schema enforces shape; the validation script additionally enforces identical field-key sets, task semantics, source quotas and target membership.

## Exact conversion rules

| Sources | Rule |
|---|---|
| MultiNLI, SNLI, ANLI, MASSIVE, BANKING77, SST2 | Keep native semantic categories. Resolve semantic targets to native option keys, deterministically reorder options, remap gold to opaque `option_N` keys. |
| RACE, SciQ, MCTest, MCScript2, ReClor, WIQA, ARC, PubMedQA artificial | Keep the native question/context and complete options. Reorder options by the same hash rule and remap the target. SciQ no longer exposes a fixed correct-at-position-zero pattern. |
| MultiRC | Boolean candidate-answer correctness, with passage and the candidate answer in context. Preserve independently labeled answers; do not force a single correct option. |
| Aegis2 | Boolean human-labeled prompt safety. Response labels and responses are absent from the model payload. |
| RuleTaker | Boolean native entailment: `entailment` maps to true; `not entailment` maps to false. Reject any other label. This corrects unsupported three-way options in the staging adapter, without inventing contradiction/unknown distinctions. |
| CivilComments | Boolean recast of the native toxicity annotation fraction: true when `toxicity >= 0.5`, including an exact-half tie. The question asks whether at least half of annotators would consider the comment toxic. Fractions remain outside the payload; they are neither confidence nor objective truth. |
| HelpSteer2 ratings | Five separate fixed-order 0–4 enum fields: helpfulness, correctness, coherence, complexity and verbosity. Preserve native levels. Definitions are frozen in code and paraphrase the [paper's annotation guidelines](https://arxiv.org/html/2406.08673v1#A7.SS3.SSS1). Complexity and verbosity describe attributes; higher values do not mean better quality. |
| HelpSteer2 preferences | Native preference sign selects response 1, response 2 or tie. Keep all three options. Retain preference strength separately as audit metadata. |
| GSM8K | Parse the native final answer as exact Decimal. Construct five numeric options using offsets `[0,-2,-1,+1,+2]`, then reorder. Gold is the zero-offset option. These distractors are synthetic nearby wrong numbers, not human error samples or model-generated reasoning. Native reasoning is absent from the input. |

The generator uses no randomly generated labels or schema descriptions. Options can contain the correct answer, as required for supervised multiple choice, but their order carries no intentional correct-position marker. Annotation labels, native rationales, gold, inverse mappings and traces are never passed to the model.

## All seeds and ordering rules

| Operation | Seed | Algorithm |
|---|---:|---|
| Native source sampling, all 20 sources | `20261014` | Python `random.Random`, pinned Python 3.14.7 and native row ordering. Operational sources use a 300-row reservoir, retaining 250 plus 50 reserves. Reading sources sample 300 indices without replacement, first 250 selected then sorted into native order. Specialist sources sample 250; HelpSteer2 samples 125 ratings then 125 training preferences with one shared RNG. |
| Specialist reserves: PubMedQA, RuleTaker, ARC, GSM8K; HelpSteer2 ratings | `20261015` | Separate RNG sampling remaining rows without replacement; 50/source, or 25 HelpSteer2 ratings. |
| HelpSteer2 preference reserves | `20261016` | Separate RNG selecting 25 remaining training preferences. |
| Typed option ordering | `20261016` | SHA-256 ordering described below; no PRNG calls. This separate operation shares a numeric seed with preference reserves but has independent inputs and algorithm. |
| Python hash seed in replay subprocesses | `0` | `PYTHONHASHSEED=0` for interpreter hashing; schema ordering does not use Python's randomized `hash()`. |

For each option at original index `i`, compute:

```text
canonical = JSON([20261016, record_id, field_name, i],
                 sort_keys=True, separators=(",", ":"),
                 ensure_ascii=False, allow_nan=False)
rank = SHA256(UTF8(canonical)).hexdigest()
```

Sort ascending by `(rank, original_index)`, assign `option_0`, `option_1`, etc., and remap gold by original key. Each trace contains the original-to-typed mapping and descriptions. Score levels are ordered semantically and are not shuffled. Output serialization is UTF-8 with sorted object keys, compact separators, no nonfinite floats, and one LF after each record. Input record order is preserved. No timestamps, host paths, process IDs, random UUIDs or mutable remote lookup affect typed corpus bytes.

## What was executed and verified

1. Froze the 26 candidate data-file URLs, sizes and SHA-256 hashes in `configs/training-raw-files.json`; verified every existing raw file. The source pin inventory is `configs/training-source-pins.json`. BANKING77's acquisition now uses the recorded Git commit, rather than asking GitHub for its latest commit.
2. Created a local `.venv` with Python 3.14.7 and the exact dependency versions in `requirements-training-data.txt`.
3. Re-executed all three native preparation scripts against the cached pinned raw data. All 20 selected files and 20 reserve files reproduced their recorded bytes exactly. Expected hashes are frozen in `configs/training-native-output-hashes.json`.
4. Re-executed integration against the 11 local evaluation corpora, verifying their hashes against `data/manifests/benchmark-preprocessing.json`. The local exact/shingle heuristic found no matches, and no reserves were consumed. Evaluation corpora were read only; no new HOLD/EXCLUDE dataset was downloaded.
5. Generated all 5,000 typed points locally using the frozen input checksum and seed. Validated every outer training-point schema and every gold target against its point's decision schema. Checked field sets, source quotas, ID uniqueness and task types.
6. Regenerated the typed corpus in a separate temporary directory and compared bytes. Tests also check choice-label preservation, critical boolean/rating recasts, separation of gold from model input, input-checksum rejection, invalid RuleTaker label rejection and Decimal-context independence.

The reproducibility and validation receipts are `outputs/training-data-reproducibility.json` and `data/training/manifests/typed-validation.json`. Source manifests retain native labels and acquisition metadata; typed manifests retain transformation hashes. Large corpora remain locally cached and excluded from Git; scripts, schemas, configs, hashes, tests and this documentation remain versionable.

## Replay commands

Run from the repository root, with CPython 3.14.7. Create the environment if needed:

```sh
uv --cache-dir /private/tmp/ac-worktrial-uv-cache venv .venv --python 3.14.7 --no-project
uv --cache-dir /private/tmp/ac-worktrial-uv-cache pip install --python .venv/bin/python -r requirements-training-data.txt
```

To regenerate and validate the typed data from the frozen native sample, without downloading or model calls:

```sh
.venv/bin/python scripts/reproduce_training_data.py
```

To also reconstruct the native sample from pinned raw candidate files, downloading only missing files from the frozen allowlist:

```sh
.venv/bin/python scripts/reproduce_training_data.py --rebuild-native --evaluation-dir /Users/bwt/ac-worktrial/data/processed
```

For raw-cache verification without downloads:

```sh
.venv/bin/python scripts/fetch_training_raw.py --verify-only
```

Reproducibility means byte-identical outputs for the recorded code, pinned bytes, versions and seeds. Any checksum drift causes failure; no replacement source or latest revision is silently accepted. Availability of remote archives is outside the generator's control, so preserve the raw cache or back it up with its checksum inventory. The full replay requires the original 11 evaluation files (or byte-identical copies at the supplied path); their exact hashes are recorded. No split assignment, expanded task variants, training run or calibration fit is performed here. The overlap heuristic remains a limited screen, not proof that all upstream contamination is absent.

## Frozen checksums

Native input SHA-256: `7ac75ce3895d89f22cc4ef9095112302454d0df3ff64c5c6cdfe9b6f86f3d80b`.

Typed output SHA-256: `3b3d8db82da7595565ad48c8cb7da981821097324c151169cf64c4c11fc3ea36`.

The replay validator checks these frozen values plus the generator and training-point schema hashes in `configs/typed-generator-config.json`. All eight tests passed.
