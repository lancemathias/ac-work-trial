# E8/E9: document length and Boolean polarity coverage

This post-hoc audit uses the checksummed original v3 text, exact E8/E9 native training membership, the fixed panel, and Experiment 9's already-selected λ=0.3 predictions. It does not change model selection. Lengths below measure serialized state characters, not tokens or full prompt length.

## Boolean coverage

All 769 native Noul training examples are MultiRC. Every one ends its instruction with “Is the candidate answer correct?” Labels include 345 True and 424 False. E8/E9 use 11,755 training views, of which these unchanged Noul examples comprise 6.54%; Choice examples receive a second permutation view. The 1,511 native two-option Choice examples come from HelpSteer2 (812) and Implicatures (699), rather than a broad set of Boolean decision predicates.

An instruction-only lexical screen finds negative words in 27 of 769 Noul questions. This does not measure semantic negation coverage: a negation can occur in an embedded reading question or candidate answer, a lexical hit may be irrelevant, and the screen misses other locations and paraphrases. False targets on affirmative questions also do not establish competence on negated predicates.

RAGTruth asks “Does the response contain any hallucinated or unsupported content relative to the source?” Thus True denotes a defect, whereas True in the MultiRC training decision denotes a correct candidate answer. TabFact and TruthfulQA use affirmative support/truth questions. This supports a task/polarity coverage hypothesis, but a controlled positive-versus-negated-predicate comparison has not been run.

## Length coverage

| State characters | Native training | Native Noul training | RAGTruth test | ContractNLI test |
|---|---:|---:|---:|---:|
| Median | 805 | 1,602 | 2,897 | 10,172 |
| 90th percentile | 2,691 | 2,478 | 5,624 | 23,166 |
| Maximum | 11,874 | 3,605 | 10,397 | 42,080 |

79 ContractNLI test examples exceed the longest native training state (11,874 characters). On this slice, Experiment 9 scores 70.89% versus matched base logits 78.48%. Within the 153 RAGTruth examples at or below the longest Noul training state (3,605 characters), Experiment 9 scores 50.98% versus base 77.78%. RAGTruth degradation therefore does not require inputs longer than every Boolean training example. These slices differ in content and labels; they do not isolate a causal length effect.

## Useful controlled check

Cross document length and evidence position with affirmative/negated predicates and Noul/True-False Choice interfaces. For a decidable presence task, pair “X is present” with “X is missing,” reverse the target, and align semantic probability mass before comparing. Keep all paired versions of a document in the same train/development/test group. Specify whether “missing” means absent from the supplied document or absent in the world; lack of mention does not establish the latter without an explicit task convention. Separate lexical negation, absence, contradiction, and insufficient evidence when the task permits it.
