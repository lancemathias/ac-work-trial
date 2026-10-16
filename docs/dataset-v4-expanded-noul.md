# v4 Boolean coverage extension

This is the current v4 text release, extending the earlier WiCE/evidence/MASSIVE release without replacing its records, cached views, or provenance. Only official training data are used. CondaQA is filtered to exact YES/NO labels **before** selecting 2,000 examples: YES maps to true, NO to false. The 647 DON'T KNOW records and 108 other free-form answers are excluded, leaving 5,077 eligible native rows before overlap screening. The original passage and question remain intact; uncertainty is never mapped to false.

| Noul source | Prior train count | Added distinct examples | New train count |
|---|---:|---:|---:|
| MultiRC | 769 | 769 | 1,538 |
| Evidence sufficiency | 1,000 | 1,000 | 2,000 |
| CondaQA | 0 | 2,000 | 2,000 |
| CREAK | 0 | 1,000 | 1,000 |

These are new examples, not repeated or reweighted records. MultiRC preserves the parent's native candidate-answer correctness representation and its 345 true / 424 false count in the added subset. Existing training paragraph families may provide additional candidate decisions, while every held-out family is excluded. Evidence sufficiency adds 250 per predicate/truth combination, with distinct groups across old and new selections. CondaQA and CREAK selection is label-blind SHA256 ranking after eligibility filtering. The selected CondaQA labels are 986 true / 1,014 false; CREAK has 485 true / 515 false. CREAK explanations are never included in model inputs.

The sources retain their original training labels and source/group IDs. CondaQA contrastive variants share their original passage family. CREAK records share entity page families. No official dev/test examples from either source were downloaded for training selection. CondaQA original Wikipedia passages are screened as well as their edited model-visible passages.

The previous WiCE (1,000 three-way Choice) and MASSIVE (1,000 60-way Choice) additions stay unchanged. The total training split is **14,031 native examples**; dev remains **809**, calibration **857**, byte-identical to the original E8/E9 splits. The training loader validates all roles, IDs, groups, states, requests and hard targets. Conversion tests cover unknown filtering, YES/NO polarity, CREAK explanation exclusion and MultiRC identity/representation.

## 27B feature preprocessing

The expanded inventory reuses all 16,755 previous v4 views and adds 4,769 Noul native views, for **21,524 training views**. There are no new Choice examples in this extension, so no new shuffled clones are necessary. Every existing Choice example still has its native view and exactly one seeded shuffled clone, including those introduced by the previous v4 stage.

New features use the exact pinned `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the existing 255-code prompt, frozen bfloat16 final-token features and extraction batch size four. All new prompts must pass the 8,192-token no-truncation check before extraction. The extension is queued after the first v4 cache finishes; it does not allocate a second GPU while that extraction runs. Its merged cache copies dev/calibration tensors and items unchanged and retains the same test reference. No head training or benchmark scoring is launched.

The host finalizer automatically downloads the completed tensor archive, verifies its SHA256/size and every archived file against the cache checksum manifest, and commits completion/export receipts. Until `local-cache-export.json` reports completion, the cache is queued or running rather than complete.

## Unweighted type balance

Each example, cached view or benchmark question counts once; benchmark questions are pooled without suite weights.

| Set | Choice | Noul | Score |
|---|---:|---:|---:|
| Native training | 7,493 (53.40%) | 6,538 (46.60%) | 0 |
| Training including Choice clones | 14,986 (69.62%) | 6,538 (30.38%) | 0 |
| Benchmark panel | 1,467 (69.49%) | 638 (30.22%) | 6 (0.28%) |

The view-level type mixture now approximates the benchmark panel's type mixture. This is a count comparison, not evidence that domains, question difficulty, label polarity or generalization now match.

## Provenance and limits

CondaQA: [official repository](https://github.com/AbhilashaRavichander/CondaQA), revision `bd4857f1f2819937f1505925ad8c809544e8e5ef`, Apache-2.0. CREAK: [official repository](https://github.com/yasumasaonoe/creak), revision `5e892b28b6dad3e87769cfc01f00a6577ca44068`, MIT. MultiRC uses the existing pinned SuperGLUE training parquet (SHA256 `ced6d731a335a0e4fbd29f6f382f291518952c0f449448361d294000c39e6728`). Evidence sufficiency uses the same pinned filtered tasksource shards as the first v4 release.

All additions are checked against the eleven full local benchmark pools and parent held-out text components using normalized exact and approximate five-word overlap/containment matching. Short CREAK claims also receive normalized exact checks at three words or more. Existing IDs and exact requests are excluded. This lexical audit cannot certify unknown paraphrases or upstream independence; parent hygiene is inherited rather than independently recertified. New sources remain training-only, so fixed dev does not directly test their source-specific generalization.

Configuration: `configs/dataset-v4-expanded-noul.json`. Full local text/splits/view plan: `data/training/v4-expanded-noul`. Durable source hashes, selected IDs/labels/provenance, screening and test results: `outputs/data-releases/v4-expanded-noul`. The active release pointer is `configs/dataset-v4-active.json`.
