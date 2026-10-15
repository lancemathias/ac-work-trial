# Matched source add-back ledger

Same-seed 9B source screens; 6,512 training, 809 dev and 857 calibration decisions; 510 updates, math head SDPA, accuracy/earliest-tie selection and calibration-only temperature. Repeatedly observed test panel; intervals omit seed variability and selection uncertainty.

Control composite: 60.9281%. Each intervention adds 250 decisions; its comparison control adds 250 unused examples from existing sources.

| Condition | Train seconds | Epoch | Dev accuracy | T | Composite | Delta pp | Target delta pp | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [ruletaker](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/df2fil2b) | 614.1 | 5 | 74.66% | 1.838 | 60.93% | +0.006 | -2.50 | no demonstrated gain |
| [aegis2](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/4acn895m) | 602.5 | 5 | 75.77% | 1.961 | 60.20% | -0.731 | -4.25 | no demonstrated gain |
| [civilcomments](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/nastxy6m) | 603.2 | 5 | 74.91% | 1.870 | 61.45% | +0.524 | +0.75 | no demonstrated gain |
| [snli-multinli](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/l4nad696) | 612.9 | 3 | 75.65% | 1.456 | 61.05% | +0.119 | +0.00 | no demonstrated gain |
| [massive](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/hqwshgo4) | 624.8 | 5 | 75.53% | 1.722 | 62.04% | +1.110 | +6.50 | advance for confirmation |
| [banking77](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/1npt61ie) | 637.6 | 5 | 74.66% | 1.694 | 60.38% | -0.547 | -4.50 | no demonstrated gain |

Primary advance threshold: +1 pp composite. Mechanism threshold: +5 pp mean on the prespecified targeted suites. Passing either warrants further investigation; neither establishes a robust source effect.

- **ruletaker**: Explicit-rule support and negation improve Boolean entailment transfer. Target: TabFact. Paired conditional 95% interval: [-0.994, +1.009] pp.
- **aegis2**: Human safety labels with True=unsafe improve instruction/polarity-sensitive Boolean judgments. Target: TabFact, TruthfulQA_binary. Paired conditional 95% interval: [-2.074, +0.631] pp.
- **civilcomments**: Toxicity-threshold supervision broadens Boolean judgments; native True scarcity may limit benefit. Target: TabFact, TruthfulQA_binary. Paired conditional 95% interval: [-0.707, +1.733] pp.
- **snli-multinli**: Cross-domain three-way entailment restores contradiction/evidence discrimination. Target: ContractNLI. Paired conditional 95% interval: [-1.154, +1.293] pp.
- **massive**: Many-option intent classification reduces option-prior dependence and improves option-count transfer. Target: Circa. Paired conditional 95% interval: [-0.178, +2.319] pp.
- **banking77**: Many-option banking classification reduces option-prior dependence and improves option-count transfer. Target: Circa. Paired conditional 95% interval: [-1.801, +0.755] pp.

All listed sources currently retain collapse on at least one diagnostic suite. See individual decision receipts for class distributions. RuleTaker has a separate preregistered instruction-semantics repair; its failed original result remains in this table.

Frozen 27B pretrained readout is a separate candidate at 77.1359%, not a matched source-screen control. Jev1.13 target: 85.82%.

Machine-readable results and input hashes: `outputs/autoresearch/source-screen-ledger.json`.
