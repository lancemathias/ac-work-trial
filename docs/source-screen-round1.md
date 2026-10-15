# Deterministic source screen: first completed conditions

The control uses accuracy checkpoint selection with earliest-epoch ties, independent calibration temperature, fixed seed 20261014, 6512 training decisions and 510 optimizer updates. Head attention uses math SDPA; the same resident frozen 9B worker and cached extraction regime serve all conditions. Two full control runs reproduce all scientific outputs exactly on this worker. This does not establish cross-seed reproducibility.

| Condition | Composite | Change versus control | Decision |
| --- | ---: | ---: | --- |
| Existing-source addition control | 60.9281% | — | Fixed reference |
| RuleTaker | 60.9343% | +0.0062 pp | No demonstrated gain |
| Aegis2 | 60.1975% | -0.7306 pp | No demonstrated gain |
| CivilComments | 61.4523% | +0.5242 pp | Below preregistered thresholds |
| SNLI + MultiNLI | 61.0467% | +0.1186 pp | No demonstrated gain |

Actual exported training artifacts pass all 18 protocol checks for each source contrast. RuleTaker's paired sampling interval is [-0.9940, +1.0086] pp; Aegis2's is [-2.0740, +0.6315] pp. These are conditional intervals on an observed panel, without seed or selection uncertainty. Neither condition meets its primary +1 pp criterion or targeted +5 pp criterion; neither triggers a matched MultiRC follow-up.

RuleTaker does not fix Boolean collapse: TabFact is 47%, with no correct False predictions. Aegis2 gives 47.5% TabFact with all True predictions and no ContractNLI Contradiction predictions. Its targeted TabFact/TruthfulQA mean drops 4.25 pp. Untargeted Finance gains do not rescue the failed criteria.

## Prospective RuleTaker semantics repair

The original recast asks whether the hypothesis follows without specifying RuleTaker's closed-world semantics. The source paper explicitly describes negation as failure: https://www.ijcai.org/proceedings/2020/0537.pdf (sections 3.3–3.4). Native labels are preserved and are not alleged incorrect. A lexical screen identifies 22 unambiguous synthetic examples and 16 additional natural-language candidates where this omission may matter; this is not evidence that it caused the failed source result.

A preregistered repair changes only the instruction prefix on the same 250 RuleTaker records. CPU reproduction verifies all data/config bytes, identical held-out sets, identical 510 optimizer windows, original labels, schema loading and actual tokenization. Maximum changed input length is 448 tokens. The first CPU release check exposed nondeterministic metadata key ordering; sorting fixes release serialization without changing scientific data or config bytes. Both receipts are retained.

The new primary criterion is at least +1 pp over the original RuleTaker condition; the targeted diagnostic is TabFact recovery of at least +5 pp, including False recall. Also report comparison with the fixed existing-source control. It requires fresh features for the 250 changed requests and has not yet run on GPU. Do not reuse ambiguous-request features.

Durable evidence: outputs/autoresearch/{ruletaker,aegis2}-{decision,protocol-verification,vs-control-paired}.json; ruletaker-cwa-{release,cpu}.json. Checksum-verified tensors remain in local outputs/model-artifacts; trace archives and checksums are versioned under outputs/traces. CivilComments, NLI, MASSIVE and BANKING77 continue in the original preregistered order.

CivilComments: paired interval [-0.7075, +1.7333] pp, 106 wins and 90 losses. TabFact improves 2.5 pp to52%, with93 False/107 True predictions; the targeted TabFact/TruthfulQA mean improves0.75 pp. RAGTruth remains200 False; ContractNLI predicts no Contradiction. No matched MultiRC follow-up is triggered. Training export and18 protocol checks passed; completed test traces exported and rescored locally. SNLI+MultiNLI follows on the same resident worker.

SNLI+MultiNLI: paired interval [-1.1542,+1.2929] pp,87wins84losses. Targeted ContractNLI remains65.5% with101NotMentioned/98Entailment/0Contradiction plus1overlength error. RAGTruth remains200False; TabFact196True/4False. Neither criterion passes; no confirmation run triggered. Training and test exported, all18protocol checks pass, test rescored locally. MASSIVE follows on the same worker.
