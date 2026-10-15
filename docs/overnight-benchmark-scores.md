# Overnight benchmark scores

[Central ledger of all completed runs and unscored conditions](performance-ledger.md).

Generated 2026-10-15T20:06:37.625293+00:00. Accuracy percentages on the shared fixed 2,111-example panel; 200 per benchmark except 111 JevBench hard. Misses count as wrong. BBH is averaged across its included subtasks; the composite equally weights the 11 benchmark scores.

## Baselines and headline overnight runs

| Benchmark | Jev 1.13 | PPLX v1 | CLEF | CLEF Flash | E4 optimized | E5 bidirectional | Native 27B | Native + reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BBH | 92.51 | 81.46 | 75.97 | 69.50 | 61.11 | 69.69 | 75.42 | 75.42 |
| Belebele | 97.50 | 96.50 | 96.00 | 95.50 | 88.50 | 93.50 | 96.50 | 96.00 |
| Circa | 79.00 | 76.50 | 74.50 | 77.50 | 60.50 | 50.00 | 73.50 | 77.50 |
| ContractNLI | 78.00 | 84.50 | 84.00 | 85.00 | 51.50 | 64.50 | 79.00 | 80.50 |
| Finance | 96.00 | 92.50 | 92.50 | 94.00 | 43.50 | 86.00 | 83.00 | 81.50 |
| JevBench hard | 72.97 | 72.07 | 74.77 | 64.86 | 61.26 | 69.37 | 72.07 | 72.97 |
| JudgeBench | 78.50 | 77.50 | 75.00 | 63.00 | 69.50 | 78.00 | 76.00 | 77.50 |
| RAGTruth | 83.50 | 77.50 | 76.50 | 65.00 | 61.50 | 41.00 | 72.00 | 76.50 |
| TabFact | 86.50 | 90.50 | 85.00 | 72.50 | 55.50 | 63.50 | 73.00 | 83.00 |
| TruthfulQA | 88.50 | 68.00 | 74.50 | 58.00 | 46.50 | 59.00 | 69.50 | 72.50 |
| WinoGrande | 91.00 | 83.00 | 95.00 | 97.50 | 74.50 | 78.00 | 78.50 | 81.00 |
| **Composite** | **85.82** | **81.82** | **82.16** | **76.58** | **61.26** | **68.41** | **77.14** | **79.49** |

E4 optimized is the accuracy-selected 27B trained head with native causality. E5 removes the full-attention causal mask only; recurrent/convolution paths remain causal. Native 27B uses the frozen pretrained answer-letter readout. Native + reverse averages semantically aligned probabilities from both option orders; it is the best completed research run.

## Source ablations beside Jev

| Benchmark | Jev 1.13 | Source control | RuleTaker | Aegis2 | CivilComments | SNLI + MultiNLI | MASSIVE | BANKING77 | RuleTaker repaired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BBH | 92.51 | 56.46 | 55.13 | 54.23 | 55.13 | 54.17 | 56.58 | 55.25 | 55.74 |
| Belebele | 97.50 | 81.00 | 79.00 | 81.50 | 80.50 | 81.50 | 82.00 | 81.50 | 81.50 |
| Circa | 79.00 | 36.00 | 37.50 | 37.50 | 38.50 | 35.50 | 42.50 | 31.50 | 36.50 |
| ContractNLI | 78.00 | 65.50 | 66.00 | 68.00 | 66.50 | 65.50 | 65.50 | 67.50 | 63.50 |
| Finance | 96.00 | 81.00 | 88.50 | 89.00 | 87.50 | 85.50 | 88.50 | 88.50 | 90.00 |
| JevBench hard | 72.97 | 47.75 | 44.14 | 45.95 | 42.34 | 46.85 | 42.34 | 41.44 | 45.95 |
| JudgeBench | 78.50 | 63.00 | 65.00 | 62.00 | 63.50 | 64.50 | 67.50 | 65.50 | 63.00 |
| RAGTruth | 83.50 | 66.00 | 66.00 | 57.00 | 66.00 | 66.00 | 65.50 | 65.50 | 66.00 |
| TabFact | 86.50 | 49.50 | 47.00 | 47.50 | 52.00 | 47.50 | 47.50 | 47.00 | 47.50 |
| TruthfulQA | 88.50 | 64.50 | 64.00 | 58.00 | 63.50 | 64.00 | 64.00 | 60.50 | 55.50 |
| WinoGrande | 91.00 | 59.50 | 58.00 | 61.50 | 60.50 | 60.50 | 60.50 | 60.00 | 60.50 |
| **Composite** | **85.82** | **60.93** | **60.93** | **60.20** | **61.45** | **61.05** | **62.04** | **60.38** | **60.52** |

The source screen uses frozen 9B, the same Medium recipe, 6,512 training decisions, 510 updates and fixed dev/calibration sets. Each addition has 250 decisions; the control adds 250 fresh examples from existing sources. Repaired RuleTaker changes only its instructions and features. These are not matched backbone/recipe comparisons against the 27B headline runs.

## Earlier runs and rejected mixture

| Benchmark | Jev 1.13 | E1 | E2 | E3 | E4 fallback | E5 + native mixture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BBH | 92.51 | 55.86 | 60.87 | 56.04 | 58.70 | 72.04 |
| Belebele | 97.50 | 81.00 | 79.00 | 80.50 | 89.00 | 94.00 |
| Circa | 79.00 | 61.00 | 34.50 | 31.50 | 62.00 | 59.50 |
| ContractNLI | 78.00 | 53.00 | 59.50 | 66.00 | 52.00 | 82.00 |
| Finance | 96.00 | 48.00 | 78.50 | 80.50 | 47.00 | 85.50 |
| JevBench hard | 72.97 | 45.05 | 47.75 | 42.34 | 58.56 | 72.07 |
| JudgeBench | 78.50 | 44.50 | 63.50 | 65.00 | 70.50 | 80.00 |
| RAGTruth | 83.50 | 66.00 | 66.00 | 66.00 | 62.50 | 62.50 |
| TabFact | 86.50 | 56.50 | 47.50 | 47.00 | 51.50 | 68.50 |
| TruthfulQA | 88.50 | 60.00 | 62.50 | 63.50 | 45.50 | 72.00 |
| WinoGrande | 91.00 | 60.00 | 62.50 | 60.00 | 73.00 | 79.50 |
| **Composite** | **85.82** | **57.35** | **60.19** | **59.85** | **60.93** | **75.24** |

E1–E3 are historical context, not overnight reruns. E4 fallback uses the earlier fallback numerical runtime; compare E5 against E4 optimized for the attention ablation. The equal E5/native mixture passed dev selection but lost test accuracy and was rejected. E2 head-size ablations have dev measurements, not completed comparable 11-benchmark test columns.

## Pending final run and interpretation

The final bounded-thinking run is excluded until its full 11-benchmark test evaluation completes. A completed dev result alone is not a benchmark score. The user requested this be the last GPU run; automatic export and GPU release are armed, and no follow-ups will launch.

These are sampled-panel exploratory point estimates. They do not establish full-dataset superiority, seed robustness or independent confirmation. CLEF and CLEF Flash each miss one ContractNLI input; the table counts it wrong consistently. Temperatures were fitted only on permitted nonbenchmark calibration examples for trained/research candidates; released baselines retain their original probabilities.

## Evidence

- [Machine-readable scores and input SHA-256 hashes](../outputs/autoresearch/overnight-benchmark-scores.json).
- [Wide CSV with every included run](../outputs/autoresearch/overnight-benchmark-scores.csv).
- [Baseline measurements](../outputs/baselines-20261014/results.md), [source ledger](source-screen-ledger.md), and [issues hypotheses and fixes](overnight-research-log.md).
