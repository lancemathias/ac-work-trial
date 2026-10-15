# Readout experiments pending after the verified 79.49% pool

The active fixed128 thinking trial runs on the resident A100. Its original dev
accuracy gate and separate reservation gate remain unchanged. Do not redeploy
or restart its worker merely to add another method.

## Seven audited demonstrations

The original shortest-only seven-example selection was rejected before any GPU
inference after semantic review found context-free prompts and questionable
entailment examples. Preserve its original artifacts as a negative preparation
result; it has no measured model accuracy.

The replacement `pretrained-native-seven-fewshots-quality-v2` keeps the seven
source/label cells and original native labels. Locked records and individual
reasons are in `fewshot-quality-v2-selection.json` and
`fewshot-quality-v2-points.json` under `outputs/autoresearch`. Exact approved
training membership, heldout group isolation, inherited whole-benchmark
exclusion, and the pinned tokenizer passed CPU checks. Every request gains
1,034 tokens; the longest final prompt is 9,203 tokens, below the 12,288 cap.
Ten helper/evaluator tests passed, including rejection without benchmark access
when dev accuracy or remaining budget fails. A label-blind GPU capacity smoke
and resident service integration are still required before inference.

The fixed dev gate is at least one percentage point above the original native
82.447466% accuracy. The exploratory test criterion is at least one point over
77.135889%; promotion additionally requires exceeding the current 79.490516%
verified champion. These are different thresholds and both must be reported.

## Selective bounded thinking

`pretrained-order-disagreement-thinking128` was preregistered before any
thinking accuracy was inspected. It uses the existing native/reverse probability
pool when semantic argmax agrees, and fixed128 thinking when they disagree.
Router ties use the smallest semantic option ID, independent of display order.
There is no learned threshold or suite-specific routing.

The rule routes 203/857 calibration decisions and 180/809 development decisions.
These counts use only the two existing model outputs. Seven routing/assessment
contracts pass. The CPU assessment requires complete thinking calibration and
dev files, checks every role ID and native target against the approved splits,
and never reads test files. It fits a separate scalar temperature using only
calibration examples. Model selection uses dev accuracy, requiring at least
one point above the pooled 84.301607%, before any candidate test inference.
The exploratory test criterion is at least one point above pooled 79.490516%.

If the pure thinking trial has already produced complete test logits, reuse
them. Otherwise the selective candidate needs only disagreement test rows,
subject to its own capacity/reservation checks after its dev gate. Failure of
the pure-thinking gate remains a recorded result; it does not change this
separately preregistered candidate's comparator or threshold. Final frozen
parameter verification remains required before promotion.

All these comparisons reuse the project panel and are exploratory rather than
independent confirmation. No benchmark labels enter demonstrations, fitting,
calibration, routing, or development selection.

The cached selective test application passed a synthetic 2,111-decision CPU
integration test, including calibrated outputs and refusal before panel access
for a failed dev gate or unverified frozen parent. Fourteen combined CPU checks
passed. This certifies the application logic, not candidate model performance.
Subset-only GPU generation remains to be integrated if the pure-thinking run
does not produce a complete verified test parent.
