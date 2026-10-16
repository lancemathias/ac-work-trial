# Experiment branches

Each numbered experiment has a dedicated branch. Existing historical and preservation branches remain available. `anchor_commit` in the machine-readable map records the experiment snapshot before this branch-policy documentation; branch tips can advance with work on that experiment.

| Experiment | Branch | Anchor |
|---:|---|---|
| 1 | `codex/experiment-1-frozen-joint-head` | `a0bf189` |
| 2 | `codex/experiment-2-v2-data` | `f663260` |
| 3 | `codex/experiment-3-v3-data` | `d4c7248` |
| 4 | `codex/experiment-4-qwen38-27b` | `9e77b62` |
| 5 | `codex/experiment-5-residual-mlp` | `366fe8a` |
| 6 | `codex/experiment-6-option-orders` | `8df478c` |
| 7 | `codex/experiment-7-pplx-recipe` | `c19c2be` |
| 8 | `codex/experiment-8-zero-output-init` | `f47d739` |
| 9 | `codex/experiment-9-residual-l2-sweep` | `c965e1e` |
| 10 | `codex/experiment-10-v4-residual-l2-sweep` | `a1c4d0a` |

The E8 branch includes its ten-epoch results, residual audit and report. Its report was cherry-picked because the report commit was originally interleaved with E9 sweep development. E9 includes its completed sweep, coverage audit and tensor-export receipt. E10 includes v4 preparation, completed 27B feature caching, sweep metrics, full audit and checksum-verified local checkpoint export.

The user owns creating and merging the Experiment 10 PR to `main`. No PR or merge is performed as part of this organization. Raw datasets and tensor archives remain local with committed checksum/provenance receipts.

Creating these local branch refs does not publish them. Remote pushes require authorization following the earlier automatic approval-review rejection.
