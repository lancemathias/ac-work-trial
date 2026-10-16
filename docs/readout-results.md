# Qwen3.8-27B readout results

[Qwen3.8-27B readout design and terminology](native-readout-design.md).

Current best verified composite: **79.4905%** from equal native/reverse semantic probability pooling. Jev1.13 target85.82%; remaining gap6.3295pp.

Dev accuracy84.3016% versus82.4475% passes the preregistered+1pp gate. GlobalT0.988723 is fitted only on857nonbenchmark calibration examples. Frozen weights unchanged; zero optimizer updates. All2111testrecords exported with checksums and rescored locally.

| Suite | Qwen3.8-27B readout | Qwen3.8-27B + reverse | Delta pp |
| --- | ---: | ---: | ---: |
| BBH | 75.42% | 75.42% | +0.00 |
| Belebele | 96.50% | 96.00% | -0.50 |
| Circa | 73.50% | 77.50% | +4.00 |
| ContractNLI | 79.00% | 80.50% | +1.50 |
| FinancialPhraseBank | 83.00% | 81.50% | -1.50 |
| JevBench_public_hard | 72.07% | 72.97% | +0.90 |
| JudgeBench | 76.00% | 77.50% | +1.50 |
| RAGTruth | 72.00% | 76.50% | +4.50 |
| TabFact | 73.00% | 83.00% | +10.00 |
| TruthfulQA_binary | 69.50% | 72.50% | +3.00 |
| WinoGrande | 78.50% | 81.00% | +2.50 |

Paired sampling delta+2.3546pp, conditional95% interval[+0.9323,+3.7227]pp;134wins83losses. Reused panel and multiple explored candidates: this is not independent benchmark confirmation.

RAGTruth True recall improves to72.06%(49/68); TabFact True recall72.63%(69/95). Their True predictions both rise to77/200, substantially reducing the original False dominance. ContractNLI still predicts Contradiction only17times; Circa retains no ProbablyNo predictions. Full class diagnostics: `outputs/autoresearch/readout-order-decision.json`.

Deployment needs two forward passes on shared pinned27B weights; original native logits were reused for this experiment. Reported per-row latency sums amortized native/reverse forward costs, excluding input preparation and scoring. Whole new reverse-pipeline runtime1014.15s includes preparation/calibration/evaluation; it is not single-query latency.

[W&B run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/6djekmw2). Exact recipe/calibration/probabilities and all attempts: local `outputs/model-artifacts/readout-native-reverse-pool-v1`.

The real thinking128 smoke passed maximum-context memory/model execution at64.18GBpeak. Its initial upfront14.0h conservative bound exceeded12.9h remaining, before any thinking accuracy measurement. A scheduling-only amendment now budgets857cal+809dev first and all2111test rows afterward only if both the unchanged accuracy gate and a fresh remaining-budget check pass. See `thinking-staged-budget-amendment.json`; numerical model, token budget, batch size, input fingerprints, safety factor and success thresholds remain fixed.
