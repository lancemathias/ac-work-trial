# Candidate reading and reasoning training preparation

Seven candidate sources were downloaded using pinned Hugging Face dataset revisions from `work/training-discovery.json`, except MCScript2 which uses the official Saarland release URL. Only candidate sources were selected. RACE uses its `all` configuration exactly once; its high/middle variants would duplicate examples. MultiRC downloads only `multirc`, not the other SuperGLUE configurations. MCTest combines distinct mc160/mc500 training stories. No benchmark evaluation partitions were intentionally selected.

The MCScript2 official ZIP contains train, development and test XML files. We cached the published archive for provenance but parsed only train-data.xml; development/test examples never enter the processed pool. The archive is pinned by content SHA-256 because its URL has no immutable revision identifier.

Each source contributes 250 uniformly sampled native training rows without replacement, using seed 20261014. An additional 50 disjoint randomly sampled rows are reserve replacements for downstream overlap filtering and are not part of the final 5,000-row mixture. Source balancing rather than corpus-size-proportional sampling gives each candidate a usable pilot allocation. Native records are preserved under `raw_native`; labels and annotations are separate from inference-facing state/question/options. No synthetic distractors, teacher judgments, schema expansions or train/calibration split assignments have been generated.

Native training pools: RACE 87,866; SciQ 11,679; MultiRC 27,243; MCTest 1,480; ReClor 4,638; WIQA 29,808; MCScript2 14,191. Each source manifest records pins, raw file SHA-256, sample row indices, labels, groups and processed SHA-256. Release license metadata is preserved rather than assumed; MCScript2 license requires checking its upstream terms. RACE's `other` license is not a declaration of unrestricted commercial use.

## Conversion and grouping

RACE, SciQ, MCTest, ReClor, WIQA and MCScript2 preserve original answer alternatives as multiple-choice questions and targets as zero-based option indices. SciQ retains its original correct answer at index 0 in this native staging representation; an eventual training converter must shuffle options deterministically to avoid position leakage. MCTest's literal `\\newline` markers are converted to newline characters in inference-facing text; raw text remains preserved.

MultiRC retains independent correctness for each candidate answer. Its binary target does not imply a single correct answer across a question's candidates. Paragraph IDs identify its groups; WIQA groups by paragraph ID; MCScript2 by document ID; other reading sources group by exact passage hash. Derived schemas and alternative answer orderings must remain in their source group for later split assignment.

## Typed schema proposals

* Choice: directly preserve native options, deterministically randomize order and update target; use opaque IDs with textual criteria. WIQA remains three-way influence classification; multiple valid MultiRC answers must not be forced into a single-choice schema.
* Noul: ask whether an independently chosen candidate answer is correct, preserving original positive/negative supervision. A choice question can supply one original answer as a candidate; sample positive and negative candidates with an explicit sampling policy. For MultiRC this is the native task. Do not confuse an incorrect answer with an unanswerable question.
* Score: these sources provide no native ordered quality rubrics. Do not transform option positions into ordinal scores or invent fractional confidence labels. Binary correctness can be encoded as endpoint values only if a documented binary rubric is desired; it supplies no intermediate score supervision.

All prepared data remains staging pending the parent audit's exact/normalized source-family overlap screening. Public training splits do not alone prove absence of shared upstream sources or contamination.
