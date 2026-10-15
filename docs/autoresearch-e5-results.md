# E5 full-attention ablation

Fixed composite: 68.4141% vs 61.2611% native. Delta +7.1530pp; conditional paired95% interval[4.6412,9.8293]pp. Single-seed exploratory result on a reused panel; not independent confirmation.424 paired gains/274 losses. One overlength ContractNLI miss remains wrong in both conditions.

| Suite | Native % | E5 % | Delta pp |
|---|---:|---:|---:|
| BBH | 61.11 | 69.69 | +8.57 |
| Belebele | 88.50 | 93.50 | +5.00 |
| Circa | 60.50 | 50.00 | -10.50 |
| ContractNLI | 51.50 | 64.50 | +13.00 |
| FinancialPhraseBank | 43.50 | 86.00 | +42.50 |
| JevBench_public_hard | 61.26 | 69.37 | +8.11 |
| JudgeBench | 69.50 | 78.00 | +8.50 |
| RAGTruth | 61.50 | 41.00 | -20.50 |
| TabFact | 55.50 | 63.50 | +8.00 |
| TruthfulQA_binary | 46.50 | 59.00 | +12.50 |
| WinoGrande | 74.50 | 78.00 | +3.50 |

Nine suites improve, but Circa drops10.5pp and RAGTruth20.5pp. RAG predicts182True/18False (True recall97.06%, False12.12%). TabFact170False/30True (False96.19%, True27.37%). ContractNLI now predicts all three classes:147Entailment/29Contradiction/23NotMentioned, though NotMentioned recall26.51% remains weak. Circa predicts93ProbablyYes but only15Yes; Middle remains absent. Finance57positive/37negative/106neutral, accuracy86%. No task-specific thresholds or prompt changes fitted to these outcomes.

Epoch2 accuracy-selected83.9308% dev; calibration-onlyT1.06454. Exact initial-head SHA and490 training windows match native; identical split bytes, backbone frozen. Training737.35s; benchmark884.63s; benchmark peak56,535,166,464bytes. Full training/splits/test exports checksum-verified. Preregistered practical gain criterion passes; additional seed pairs required before a robust-effect claim. Existing source-grid/polarity/readout queue proceeds unchanged.
