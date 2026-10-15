# Experiment 8: residual correction inspection

Inspect the selected checkpoints of the five-epoch run (epoch 1) and ten-epoch run (epoch 8), using all 2,111 benchmark examples. Subtract saved matched-base logits from trained logits, align semantic option IDs exactly, retain only valid options, and remove each correction vector's mean. A common logit shift has no predictive effect. Probabilities are uncalibrated to isolate the head from temperature fitting. This is descriptive analysis of a repeatedly observed panel, not new model selection or independent confirmation.

| Measure | Five-epoch run | Ten-epoch run |
|---|---:|---:|
| Median max-minus-min correction, logits | 0.533 | 1.249 |
| Centered entries smaller than 0.01 in magnitude | 13.2% | 2.6% |
| Centered entries smaller than 0.1 in magnitude | 31.6% | 26.1% |
| Examples with probability total variation below 0.01 | 25.4% | 19.1% |
| Base argmax changed | 9.4% | 19.7% |
| Correct → wrong | 84 | 252 |
| Wrong → correct | 97 | 135 |

The corrections are generally dense by small-magnitude thresholds, and operate across many examples rather than only a small exception set. They can still be concentrated within an example: among the 200 eight-option examples, the median share of correction energy in the two largest centered entries is 87.6% / 87.9%. All eight-option examples belong to Circa, so this does not establish a universal option-count effect. In binary examples (1,110/2,111), centering necessarily produces equal-and-opposite entries; top-k coordinate sparsity there is uninformative. Magnitude thresholds are illustrative, not established semantic cutoffs.

A more striking finding is near-constant slot bias within some suites. On RAGTruth, the ten-epoch mean centered correction is [+0.940, -0.940], explaining 94.2% of squared correction energy. This shifts the first-versus-second log odds by about +1.88 on average. The head flips 80.5% of RAGTruth answers, breaking 118 previously correct answers while fixing 43. For ContractNLI, the mean slot vector explains 88.1% of correction energy. These are descriptive mean-vector decompositions; they do not prove whether the model learned semantic label priors, display-position priors, or both.

These results support controlling correction magnitude and checking label/position bias. They do not, by themselves, establish that L1 is preferable: sparse corrections can still be confidently wrong. No weight or hidden-activation sparsity was measured.
