# Tasksource-filtered candidate shortlist

Scope requested by the user: dataset descriptions and only the first two stored train rows per candidate in the local filtered Parquet snapshot. Shards are ordered by filename; row order is preserved. The first two rows often share a group and are recasts, not independent examples. No performance experiment or full contamination audit was run. Source-column inventory was used only to identify available candidates. Current upstream dataset cards can describe revisions newer than the frozen filtered snapshot.

## First mix

| Source | Fit | Evidence from the first two train rows |
|---|---|---|
| wice | Natural-document grounding and support/partial-support/unsupported decisions | Two distinct evidence/claim cases, 2,581 and 13,776 state characters. First is unsupported; second is supported. The second claim is unusually trivial, so these examples do not certify general annotation quality. |
| CONDAQA | Paragraph-based negation with explicit unknown | Both rows are the same paragraph/question under an instruction paraphrase; answer is DON'T KNOW. The dataset description specifically targets implications of negation. Keep YES/NO/UNKNOWN distinct. |
| help-nli | Negation scope and monotonicity | A packed example contrasts an unsupported expansion of a claim with an inference from no radios to no superheterodyne receivers. The first row classifies one item; the second is Noul comparing the two labels. Same 381-character state, not two independent cases. |
| procedural-typed-decisions/evidence_sufficiency | Direct Noul, insufficient evidence and conflict detection | First asks whether a claim has sufficient support; duplicate evidence origins do not count as independent. Second asks whether both support and contradiction exist. Both targets are false in the same 718-character state. Explicit rules make the synthetic semantics inspectable. |

## Useful auxiliary candidates

- `procedural-typed-decisions/needle_retrieval`: Choice lookup plus direct Noul verification on the same 2,028-character shipment register, including reissued-ID aliases. Useful retrieval/format exercise. The first two rows alone do not demonstrate long-document coverage or a missing-ID example.
- `babi_nli/simple-negation`: explicit negative/location-update context; first row is non-entailment and second verifies that label via Noul. Useful small controlled supplement, not natural long-document grounding.
- `procedural-typed-decisions/policy_applicability`: explicit defaults and the semantics of an omitted constraint. The first row is ordinal risk, second is Noul access denial. A secondary fit for rule/absence semantics.

## Defer or reserve

- `doc-nli`: semantically attractive, with Choice and Noul non-entailment queries in the first packed example. However, DocNLI's source description includes CNN/DailyMail, which also supplies RAGTruth inputs. This is a shared-corpus risk, not proof of a duplicate. Hold pending article-level provenance if avoiding RAGTruth overlap is the priority.
- `nan-nli`: the original card explicitly describes 258 evaluation samples, even though Hugging Face exposes a split called train. Reserve as a negation diagnostic rather than automatically treating the split name as training provenance.
- `procedural-typed-decisions/partial_observation_calibration`: first two Noul targets are 0.25 and 0.157895. These are genuine soft probabilities; current hard-target CE cannot consume them faithfully without a soft-target objective. Not the immediate Boolean-polarity addback.
- `procedural-typed-decisions/policy_under_uncertainty`: useful uncertainty semantics, but potentially distributional targets and a more complicated objective than the immediate task.
- `ruletaker`: first inspected row appears inconsistent: a fact and rule derive a positive chase relation, while the negated hypothesis receives entailment. Resolve this example before accepting the slice; this is not a claim that every row is defective.
- `lonli`, `vitaminc`, `record_aggregation`: potentially useful broader reasoning/verification supplements, but weaker immediate evidence for the specific long-document and negated-predicate gaps.

## Contamination interpretation

The shortlist avoids direct use of our eleven evaluation dataset families. A description and two examples cannot establish zero overlap or unknown upstream derivations. The filtered-full card describes training-only selection and protected text/source checks, but explicitly limits universal decontamination claims. Keep the existing whole-benchmark exclusions and group related recasts/underlying documents together. This limited review does not replace those controls or certify the candidate sources as clean.

Descriptions consulted:
- Filtered snapshot policy: https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/main/filtered-full/README.md
- CONDAQA: https://arxiv.org/abs/2211.00295
- HELP-NLI: https://huggingface.co/datasets/tasksource/help-nli
- WiCE: https://huggingface.co/datasets/tasksource/wice
- Procedural description: https://github.com/sileod/tasksource (typed-decision section)
- NaN-NLI: https://huggingface.co/datasets/joey234/nan-nli
- DocNLI dataset description: https://aclanthology.org/2021.findings-acl.435.pdf
- RAGTruth dataset description: https://aclanthology.org/2024.acl-long.585.pdf
