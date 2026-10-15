# Benchmark access and preprocessing

All 11 benchmark sources were downloaded without authentication and normalized
locally on October 14 2026. This is an explicit local evaluation protocol, not a
verified reproduction of Perplexity's reported panel. No inference or GPU jobs
were run. No evaluation examples were assigned to training or calibration.

## Current outputs

| Suite | Decisions | Local conversion |
| --- | ---: | --- |
| WinoGrande | 1,267 | XL labelled validation, two choices |
| FinancialPhraseBank | 2,264 | All-agree subset, three sentiment classes |
| RAGTruth | 2,700 | All test responses; any annotated hallucination span means true |
| JudgeBench | 350 | GPT-4o response cohort; select more correct response A or B |
| BBH | 5,508 | Explicit multiple-choice and Boolean tasks; retain other tasks raw |
| JevBench public hard | 111 | Preserve 67 Choice, 38 Noul and 6 Score questions |
| TabFact | 12,779 | Full labelled test, statement supported by table yes/no |
| ContractNLI | 2,091 | Full test contracts; each hypothesis is a three-way NLI decision |
| Circa | 31,525 | Source-wide fine-grained majority labels; exclude 2,743 NA rows |
| Belebele | 900 | English only, four choices |
| TruthfulQA binary | 5,882 | Each MC2 answer is a binary truth decision; 817 question groups |

Total: **65,377 decisions**. Circa's full source is not an official held-out test
split; it is designated evaluation-only in this project. Its NA means no majority
annotation, not a semantic answer category. Gold fields, human rationales and
evidence-span annotations are excluded from the inference payload.

## Reference comparison limits

The PPLX card names these suites but does not establish the exact subsets, prompts
or conversions needed for an identical panel in the source files inspected. Its
released `source/src/autojev/data.py` uses a different task list; do not equate
those partitions with this 11-suite evaluation.

Resolve agreement subset, response cohort, BBH task selection, Circa label
granularity, Belebele language, TruthfulQA binary conversion, and sampling before
claiming a matched PPLX result. RAGTruth here measures response-level accuracy,
not span-level or hallucination F1. ContractNLI here uses three-way accuracy.

BBH retains all 27 source subtasks raw. The normalized set excludes 1,000
free-form examples from Dyck languages, arithmetic, object counting and word
sorting. It also excludes three examples whose targets do not match their listed
option keys (one movie recommendation and two ruin names). Exclusion counts are
explicit in the manifest. Do not silently invent distractors or reinterpret
these records. Report accuracy per included subtask and macro-average across
subtasks before averaging BBH with the other ten suite scores.

JevBench confirms that ordered Score is present in the evaluation set: preserve
the distribution-based score interface, and document whether its hard-label
metric uses the most likely level or rounding the expected score. Preprocessing
retains the gold zero-based level; it does not choose that scoring rule.

## Reproduction and provenance

Install `requirements-preprocess.txt` in a local virtual environment, then run
`scripts/preprocess_benchmarks.py`. Run
`python -m unittest discover -s tests -v` with the same environment for artifact
checks. The pipeline reads pinned GitHub/Hugging Face revisions, never executes
remote dataset scripts, and needs no model weights or API credentials.

- `data/raw/`: cached source files, including archive contents not selected for
  evaluation. Raw RAGTruth downloads include training rows; the normalized output
  includes test rows only.
- `data/processed/<suite>.jsonl`: evaluation-only normalized records. Send only
  `state` and `question` to the model, never the entire record with its gold label.
- `data/manifests/benchmark-preprocessing.json`: exact URLs, source and output
  SHA-256 hashes, byte counts, label distributions, groups, conversion notes and
  script hash. Reruns reject altered cached source files.

Related examples share group IDs for cluster-aware uncertainty estimates.
TruthfulQA answers share the original question; RAGTruth responses share their
source; ContractNLI hypotheses share their contract; TabFact statements share
their table. Circa uses its ten contexts as conservative groups. No source is
silently truncated or downsampled. Token-length validation remains a later,
tokenizer-specific step.

Raw and processed corpora are ignored by Git. The script, manifest, documentation
and checks are suitable for version control; preserve the local artifacts for
evaluation. The manifest is the run receipt, not a guarantee of matched vendor
evaluation or complete training-set contamination screening.

## Sources and usage terms

| Source | Location | Terms to account for |
| --- | --- | --- |
| WinoGrande | [AllenAI dataset](https://huggingface.co/datasets/allenai/winogrande) | Check upstream release license before redistribution; HF metadata inspected did not declare one |
| FinancialPhraseBank | [Dataset card](https://huggingface.co/datasets/takala/financial_phrasebank) | CC BY NC SA 3.0; noncommercial restriction requires attention for commercial use |
| RAGTruth | [Original corpus](https://github.com/ParticleMedia/RAGTruth) | Review repository and underlying source-corpus terms before redistribution |
| JudgeBench | [Original benchmark](https://github.com/ScalerLab/JudgeBench) | Review benchmark and underlying task-source terms |
| BBH | [Dataset mirror](https://huggingface.co/datasets/lukaemon/bbh) | Review original BIG-Bench and task-specific terms; inspected mirror metadata did not declare a license |
| JevBench | [Public hard source](https://github.com/fstandhartinger/jevbench) | Public hard provenance declares MIT |
| TabFact | [Dataset export](https://huggingface.co/datasets/wenhu/tab_fact) | Dataset card declares CC BY 4.0 |
| ContractNLI | [Original archive](https://github.com/stanfordnlp/contract-nli) | Archive includes LICENSE and TERMS; review both before redistribution |
| Circa | [Original corpus](https://github.com/google-research-datasets/circa) | README says CC BY 4.0 but links to CC BY SA 4.0; resolve this discrepancy for redistribution |
| Belebele | [Original dataset](https://huggingface.co/datasets/facebook/belebele) | CC BY SA 4.0 |
| TruthfulQA | [Dataset card](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Apache 2.0 according to dataset metadata |

Terms above are an access inventory, not a completed license audit. This step
keeps corpora local and does not publish or upload them.
