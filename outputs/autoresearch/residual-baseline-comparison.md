# Residual readouts versus baselines

Accuracy (%) on the same 2,111-example, 11-suite panel. Missing predictions count as wrong; BBH averages its sampled subtasks, and Macro equally weights the 11 suites.

Fixed/permuted describe training data; both heads are evaluated in native order. Readout + rev pools semantically aligned native/reversed probabilities at inference. E4 is optimized native-causal joint-head training; E5 is its partial-bidirectional counterpart.

| Benchmark | Jev | PPLX | CLEF | Flash | E4 | E5 | Readout | Readout + rev | Fixed v3 | Permuted v3.1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBH | 92.51 | 81.46 | 75.97 | 69.50 | 61.11 | 69.69 | 75.42 | 75.42 | 69.93§ | 71.98§ |
| Belebele | 97.50 | 96.50 | 96.00 | 95.50 | 88.50 | 93.50 | 96.50 | 96.00 | 96.00 | 95.50 |
| Circa | 79.00 | 76.50 | 74.50 | 77.50 | 60.50 | 50.00 | 73.50 | 77.50 | 63.50 | 83.50 |
| ContractNLI | 78.00 | 84.50 | 84.00 | 85.00 | 51.50† | 64.50 | 79.00 | 80.50 | 69.50 | 70.50 |
| Finance | 96.00 | 92.50 | 92.50 | 94.00 | 43.50 | 86.00 | 83.00 | 81.50 | 67.50 | 78.50 |
| JevBench hard | 72.97 | 72.07 | 74.77 | 64.86 | 61.26 | 69.37 | 72.07 | 72.97 | 71.17 | 77.48 |
| JudgeBench | 78.50 | 77.50 | 75.00 | 63.00 | 69.50 | 78.00 | 76.00 | 77.50 | 77.50 | 76.00 |
| RAGTruth | 83.50 | 77.50 | 76.50 | 65.00† | 61.50 | 41.00‡ | 72.00‡ | 76.50 | 34.50† | 45.50‡ |
| TabFact | 86.50 | 90.50 | 85.00 | 72.50 | 55.50 | 63.50 | 73.00 | 83.00 | 78.50 | 83.00 |
| TruthfulQA | 88.50 | 68.00 | 74.50 | 58.00 | 46.50 | 59.00 | 69.50 | 72.50 | 74.00 | 77.50 |
| WinoGrande | 91.00 | 83.00 | 95.00 | 97.50 | 74.50 | 78.00 | 78.50 | 81.00 | 72.50 | 75.50 |
| Macro | **85.82** | **81.82** | **82.16** | **76.58** | **61.26** | **68.41** | **77.14** | **79.49** | **70.42** | **75.91** |

† Strong collapse/class-elimination evidence. ‡ Strong prediction skew worth investigating. § Small BBH subtask flags, not whole-suite collapse.

## Collapse diagnostics

- RAGTruth fixed: True on 199/200 (99.5%), despite only 68/200 True golds; False recall 1/132 (0.76%). Strong collapse.
- RAGTruth permuted: True on 167/200 (83.5%); False recall 28/132 (21.21%). Improved but strongly skewed, and 45.5% accuracy remains below the 66% always-False baseline.
- RAGTruth native readout: False on 186/200 (93%); True recall 13/68 (19.12%). Native + reverse: False on 123/200 (61.5%); True/False recalls 72.06%/78.79%. Pooling substantially reduces this bias.
- RAGTruth E5: True on 182/200 (91%); False recall 16/132 (12.12%).
- RAGTruth CLEF Flash: False on 198/200 (99%); True recall 0/68. No other whole-suite >=90% dominance or zero recall on a class with at least 10 gold examples was found for Jev, PPLX, or CLEF.
- ContractNLI E4: no NotMentioned predictions; 0/83 recall. This class elimination is absent from both new heads.
- BBH tracking_shuffled_objects_five_objects: both new heads predict A on all 8 examples, with 2/8 correct. Fixed also predicts No on all 9 navigate examples (2/9 correct); permuted recovers navigate to 9/9. Sports-understanding is all-no in both, but gold is already 7/8 no. Small slice sizes limit these conclusions.
- No other whole-benchmark >=95% dominant-label flag in the new heads. Circa improves largely through plain-Yes recall, 39/88 to 86/88; fixed overpredicts Probably yes (67 predictions for 10 golds). Rare Circa labels still have zero recall (support only 1–2).

## Interpretation

Permutation augmentation raises Macro 70.42% → 75.91% (+5.49 pp), but remains below pretrained Readout (77.14%) and Readout + rev (79.49%). Gains are largest on Circa (+20 pp), Finance (+11 pp), and RAGTruth (+11 pp); the latter remains a failure. The same five-epoch recipe yields 920 updates on v3.1 versus 490 on v3, and doubles choice-view weighting relative to noul. This one-seed contrast does not isolate permutation augmentation from update count and weighting.
