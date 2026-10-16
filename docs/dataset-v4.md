# v4: WiCE, evidence sufficiency and MASSIVE

This is the initial v4 stage. The current v4 release is documented in [the Boolean coverage extension](dataset-v4-expanded-noul.md), which preserves this stage and adds further Noul examples.

This release appends 3,000 training examples to the E8/E9 native v3 data, then amends the v3.1 frozen-feature view inventory. Existing dev (809) and calibration (857) files are byte-identical. It does not train a head or change the benchmark panel.

| Addition | Native examples | Type | Options | Extra shuffled views |
|---|---:|---|---:|---:|
| WiCE | 1,000 | Choice | 3 | 1,000 |
| procedural-typed-decisions/evidence_sufficiency | 1,000 | Noul | 2 | 0 |
| MASSIVE | 1,000 | Choice | 60 | 1,000 |

WiCE retains supported, partially_supported and not_supported as distinct options, balanced 333/333/334. Evidence sufficiency retains native Boolean labels, with 250 examples for each combination of claim_supported/has_conflict and false/true; one question per evidence group. MASSIVE retains the exact 250 overnight examples and appends 750 direct training examples from the local filtered tasksource pool, preserving all 60 intent meanings. The overnight gain supports only the original 250-example condition; the expanded mixture has not been evaluated.

The parent v3.1 contains no dataset with more than five options. Native v4 train has 9,262 examples. Its feature inventory has 16,755 views: 11,755 reused parent views and 5,000 new views (3,000 native plus 2,000 shuffled Choice clones). Each new Choice receives exactly one epoch-1 permutation under seed 20261014, preserving its semantic target. Chance identity permutations are retained, matching v3.1. Noul gets no shuffled clone.

All features use the pinned Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the original 255-code prompt and final-token extraction, batch size four, bfloat16 frozen backbone, and the existing optimized runtime. The cache builder validates the parent's backbone, revision and codebook. It copies dev/calibration tensors and items unchanged and retains the parent's test reference. Tokenizer preflight checks every new native and shuffled prompt against 8,192 tokens without truncation before starting extraction.

## Unweighted type balance

Every question/view counts once. Benchmark proportions pool the 2,111 panel questions, without suite macro-weighting.

| Set | Choice | Noul | Score |
|---|---:|---:|---:|
| Native train | 7,493 (80.90%) | 1,769 (19.10%) | 0 |
| Training views including clones | 14,986 (89.44%) | 1,769 (10.56%) | 0 |
| Benchmark panel | 1,467 (69.49%) | 638 (30.22%) | 6 (0.28%) |

Choice cloning increases the type imbalance: Noul accounts for 10.56% of sampled views versus 30.22% of benchmark questions. This is a coverage observation, not a request to reweight training.

## Reproducibility and checks

[Configuration](../configs/dataset-v4-wice-evidence-massive.json), [text builder](../scripts/prepare_v4_addbacks.py), [cache builder](../scripts/cache_v4_readout.py), and [Modal entrypoint](../scripts/modal_cache_v4.py). Durable receipts are in `outputs/data-releases/v4-wice-evidence-massive`; local text artifacts are in `data/training/v4-wice-evidence-massive`, with hashes in the tracked receipts. Full tensor exports must be downloaded and checksum-verified locally before any Modal cleanup.

New text additions were screened against all eleven complete local benchmark pools and parent held-outs, using normalized exact matching and approximate five-word shingle overlap/containment, plus parent ID/group/context checks. No benchmark labels enter selection. The release inherits the frozen parent's recorded hygiene; this is not a new independent audit of parent provenance. Lexical screening cannot certify absence of unknown paraphrases or upstream derivations. New sources are train-only, so fixed dev does not directly measure their source-specific generalization.

The actual training loader passes, held-out bytes match, and conversion tests cover three-way partial support, evidence grouping, rejection of fractional Boolean targets, exclusion of derived variants, and Unicode-safe JSONL serialization. See tokenizer-preflight and cache export receipts for execution status; a text release alone does not mean extraction has completed.
