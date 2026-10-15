# Candidate training data: initial 5,000-record sample

The candidate download is complete. This is a balanced pilot: 250 randomly sampled native training records from each of 20 candidate sources, seed 20261014. It is not a population-proportional random sample of the union. Source balancing prevents very large toxicity and multilingual corpora from dominating this small pilot. HelpSteer2 allocates 125 records to ratings and 125 to preference comparisons. Other combined configurations are sampled from their pooled eligible training records.

## Scope and artifacts

Only sources marked candidate in the preceding audit were processed. No HOLD or EXCLUDE source was added. Fresh procedural policy/numeric examples are a proposed generation method, not an existing dataset download, and are not counted. The initial pilot does not train a model or compute benchmark baselines.

The working corpus is `data/training/sample-5000.jsonl`. Individual screened files are in `data/training/screened/`; original selected records and 50 disjoint reserve records per source are in `processed/` and `reserve/`. Native downloaded files are preserved in `raw/`. Large data files are ignored by Git; scripts, this document, and provenance manifests remain versionable. Back up the corpus and raw caches with the manifest before deleting this worktree.

`data/training/manifests/<source>.json` records source revisions or official archive URLs, SHA-256 hashes, eligible pool sizes, sampling details, label distributions and grouping information. HF exports are pinned separately from native repository revisions where applicable. Reproduce acquisition with the three `scripts/prepare_training_*.py` scripts; run `scripts/finalize_training_sample.py --evaluation-dir /Users/bwt/ac-worktrial/data/processed` for integration and overlap screening. Sampling is deterministic for the pinned row ordering and seed.

| Source | Training relevance | Typed output methods |
|---|---|---|
| MultiNLI, SNLI, ANLI | Inference; partial relevance to ContractNLI and reasoning tasks, different domains | Native 3-way Choice; deterministic Noul per entailment criterion |
| MASSIVE, BANKING77 | Operational intent classification; partial JevBench relevance | Native intent Choice; one-vs-rest Noul |
| CivilComments | Moderation, partial JevBench relevance | Native annotation fractions as continuous scores; thresholded Noul only with explicit threshold |
| Aegis2 | Content safety, partial JevBench relevance | Human prompt safety Noul or binary Choice |
| SST2 | Sentiment; related task to FinancialPhraseBank but different domain | Native binary Choice/Noul; no justified ordinal intensity score |
| RACE, SciQ, MCTest, MCScript2, ReClor | Reading/reasoning, related task structure to Belebele/BBH | Native Choice; Noul for each proposed answer |
| MultiRC | Passage-based answer validation | Native Noul; multiple answers may be correct, so do not force single-choice |
| WIQA | Counterfactual process reasoning | Native Choice; explicit criterion Noul |
| HelpSteer2 | Response judging; relevance to JudgeBench and JevBench scores | Native rubric-specific 0–4 scores; native pairwise Choice; explicit tie handling |
| PubMedQA artificial | Biomedical evidence inference; domain differs from the 11 | Native yes/no/maybe Choice; criterion-specific Noul |
| RuleTaker | Formal inference; relevance to BBH-style reasoning | Native binary entailment Noul; false includes non-entailment |
| ARC | Science reasoning; partial reasoning relevance | Native Choice; answer correctness Noul |
| GSM8K | Arithmetic; partial BBH/JudgeBench math relevance | Generate and verify numeric candidate answers for Noul/Choice; numeric answer is not an ordinal score |

None of these sources establishes coverage of financial news sentiment, legal NDAs, table verification, conversational implicature or multilingual reading in the same distribution as the original 11 tasks. Sample manifests describe the observed distributions; balancing does not make them representative of the benchmark mixture.

## Processing and limitations

Only training records enter the eligible pools. SNLI invalid labels are removed. PubMedQA uses the artificial configuration only; the small human-labeled configuration is not downloaded. MASSIVE uses the multilingual all_1.1 export once, with translations grouped by original ID. RACE uses its combined export once; GSM8K uses main rather than duplicating socratic. Aegis uses human prompt labels with prompt-only input. CivilComments keeps annotation fractions without an invented threshold. HelpSteer2 preferences filter the native bundled file to training records. Official archives can contain validation/test files; these remain only in raw archives and never enter the sample.

Native labels, annotations and rationales are preserved in `raw_native`, separately from inference inputs. The corpus currently retains the source adapters' envelope conventions: reading/operational records use state/question/options; specialist records use input.state/input.question/input.options. Build an explicit inference serializer selecting these fields; never serialize entire records or raw_native into model input.

The integration script compares candidate input fields against state fields from all 11 locally prepared evaluation corpora. It checks exact normalized text of at least eight words, and high containment of sixteen-word shingles. Matched rows are replaced in deterministic reserve order. The integration manifest records evaluation file hashes, rejected IDs, matches, final IDs and output hashes. This screen cannot establish absence of paraphrased, translated, unobserved upstream or model-pretraining contamination. Known excluded source families stay excluded regardless of exact-match results. No claim is made about Perplexity's unpublished benchmark subsets or checksums.

## Proposed typed-schema workflow

1. Convert native semantics first. Emit Noul as a named criterion and boolean target; Choice as an explicit keyed option set and target key; Score as a documented rubric and valid numeric range. Resolve semantic NLI label targets to their native option keys. Do not infer schemas from field types alone.
2. Shuffle answer positions deterministically per family and schema seed, then remap targets. SciQ currently preserves the correct answer in position zero; this must be changed before model training. Preserve multi-answer and tie semantics rather than inventing a unique answer.
3. Use Nimble or another generator to draft schema descriptions and rubrics, then validate against native targets. Generated alternative options require independent factual or executable verification. Freeze generator model/version, prompts, seeds, schema IDs and validation traces. Keep native and generated supervision distinguishable.
4. Generate Noul variants from existing options without changing truth conditions. For arithmetic, validate generated candidate answers against the native numeric result. Scores should come from actual ordinal annotations or explicit executable rubrics; a category ID or math answer is not a score.
5. Partition by family before expansion. Keep every passage, table, translation, prompt-response family and schema variant in one partition. SNLI premise hashes and SST2 text hashes are imperfect surrogates for original image/phrase parentage. Use a separate calibration partition, and do not calibrate against the 11 evaluation suites.
6. Obtain Choice probabilities from normalized per-option logits; Noul from binary logits; ordinal scores from a calibrated categorical/ordinal distribution whose expectation yields the score. Native label agreement fractions are supervision signals, not automatically calibrated model confidence. Fit temperature or other calibration on held-out families and report log loss, Brier score and reliability bins by source/task, including imbalance and distribution shifts. Keep hard correctness labels distinct from uncertainty targets.

This acquisition step performed no schema expansion, train/calibration split, probability calibration or model training. A subsequent deterministic typed conversion is now documented in [typed-training-generator.md](typed-training-generator.md); the native sample is preserved. The 5,000 count refers to native datapoints, not multiplied schema variants.

## Verified pilot result

The combined sample contains exactly 5,000 unique record IDs, 250 per source. The local heuristic screen found zero matches against the 11 prepared evaluation corpora, so no reserve replacements were needed. This is a limited overlap result, not proof of contamination freedom. Raw caches occupy approximately 1 GB.

Combined SHA-256: `7ac75ce3895d89f22cc4ef9095112302454d0df3ff64c5c6cdfe9b6f86f3d80b`.

| Source | Eligible training records | Selected |
|---|---:|---:|
| ANLI | 162865 | 250 |
| ARC | 3370 | 250 |
| Aegis2 | 30007 | 250 |
| BANKING77 | 10003 | 250 |
| CivilComments | 1804874 | 250 |
| GSM8K | 7473 | 250 |
| HelpSteer2 | 29001 | 250 |
| MASSIVE | 598728 | 250 |
| MCScript2 | 14191 | 250 |
| MCTest | 1480 | 250 |
| MultiNLI | 392702 | 250 |
| MultiRC | 27243 | 250 |
| PubMedQA | 211269 | 250 |
| RACE | 87866 | 250 |
| ReClor | 4638 | 250 |
| RuleTaker | 480152 | 250 |
| SNLI | 549367 | 250 |
| SST2 | 67349 | 250 |
| SciQ | 11679 | 250 |
| WIQA | 29808 | 250 |
