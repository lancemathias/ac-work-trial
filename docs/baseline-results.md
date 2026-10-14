# Sampled baselines — findings

Completed on 2026-10-14. Four released decision models were evaluated on the same frozen 2,111-decision panel. No training or calibration fitting occurred. This independently defined sampled protocol is not a reproduction of Perplexity’s undisclosed panel.

| Model | 11-suite macro accuracy | Valid predictions |
|---|---:|---:|
| PPLX v1-27B | 81.82% | 2111/2111 |
| CLEF-flash | 76.62% / 76.58% | 2110/2111 |
| CLEF | 82.20% / 82.16% | 2110/2111 |
| Jev 1.13.0 | 85.82% | 2111/2111 |

Dual scores appear only for misses: exclude misses / count misses as wrong. BBH is averaged across its included subtasks; overall equally weights 11 benchmarks. Both CLEF variants rejected ContractNLI:58:nda-8 at the common 8,192-token ceiling. PPLX’s native prompt fit that input. Jev used its hosted context limits. No input was silently shortened.

Jev has the highest observed macro accuracy on this panel. CLEF leads JevBench hard, PPLX leads TabFact, and CLEF-flash leads WinoGrande. These are descriptive point estimates, not significance claims. FinancialPhraseBank uses all-agree examples, Belebele is English-only, JudgeBench uses one ordering, and Circa is a source-wide sample rather than a canonical held-out split. All assumptions are in baselines.md.

Calibration tables report NLL, multiclass sum Brier and 15-bin ECE on valid predictions. PPLX uses its released temperature 2.207568021892729; CLEF variants use released softmax; Jev returns hosted probabilities. Eight Jev predictions assign reported zero probability to the gold label. NLL includes a 1e-3 floor sensitivity check against the primary 1e-12 floor. No benchmark labels were used for recalibration. Small samples and correlated examples limit calibration conclusions.

| GPU run | Function minutes including loading | GPU-only estimate |
|---|---:|---:|
| pplx | 18.19 | $0.76 |
| clef-flash | 8.97 | $0.37 |
| clef | 20.19 | $0.84 |

Completed-function GPU-only estimate: $1.97, at Modal’s published A100-80GB rate of $0.000694/second ([Modal pricing](https://modal.com/pricing)). This is not an invoice: CPU, memory, startup/idle, the aborted import-error launch and Jev API charges are excluded. All successful GPU runs finished below their one-hour caps. PPLX and CLEF-flash ran concurrently; CLEF ran afterward with cost approval.

Local batch-one latency includes encoding and synchronization; Jev latency includes network and retries. Reference convolution/linear-attention kernels were used. These are not optimized-serving measurements.

All four runner hashes match source in their recorded Git commits. Prediction IDs, labels and panel checksums were validated before scoring. Configs, traces, receipts, result tables and calibration metrics are in outputs/baselines-20261014. Dataset text stays out of Git; source links, immutable revisions and checksums are committed.
