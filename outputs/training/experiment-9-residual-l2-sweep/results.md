# Experiment 9: Residual L2 sweep

All fits use E8 initialization, seed, cached data and ten-epoch recipe. λ penalizes mean squared centered residual logits. Select checkpoints by maximum development accuracy with earliest ties; choose λ by development accuracy, then NLL, then smaller λ. Test only the locked winner.

| λ | Selected epoch | Dev accuracy | Dev NLL | Dev correction RMS |
|---:|---:|---:|---:|---:|
| 0 | 8 | 85.17% | 0.3912 | 0.7083 |
| 0.003 | 5 | 85.29% | 0.3932 | 0.5306 |
| 0.01 | 5 | 85.17% | 0.3963 | 0.4706 |
| 0.03 | 8 | 85.29% | 0.3887 | 0.5110 |
| 0.1 | 8 | 85.66% | 0.3873 | 0.3582 |
| 0.3 | 8 | 85.78% | 0.3891 | 0.2113 |
| 1 | 9 | 85.29% | 0.3962 | 0.1045 |
| 3 | 2 | 84.43% | 0.4007 | 0.0660 |
| 10 | 1 | 83.81% | 0.4087 | 0.0053 |

Winner: **λ=0.3, epoch 8**. Benchmark macro accuracy: **76.78%**, compared with unregularized E8 **73.52%** and matched frozen code logits **78.95%**. Single-seed exploratory results on repeatedly observed panels.

[W&B group](https://wandb.ai/lmathias/ac-worktrial-decisions/groups/experiment-9-residual-l2-sweep). [Winning run](https://wandb.ai/lmathias/ac-worktrial-decisions/runs/8wzgubyg).

The zero control reproduced all E8 development accuracies and NLL exactly. No model checkpoints were published to W&B. Full tensor archive is exported separately with SHA256 verification.

Post-hoc descriptive breakdown: RAGTruth accuracy rises from 38.5% (unregularized E8) to 52.5% (λ=0.3), but remains below the matched base at 76.0%. Across the other ten suites, macro accuracy is 79.2135% versus 79.2427% for the base (a difference of -0.029 percentage points). RAGTruth consequently accounts for almost all of the remaining overall macro deficit. This observation did not enter selection.

Full archive export completed and SHA256-verified locally: `/Users/bwt/ac-worktrial/outputs/model-artifacts/experiment-8-residual-l2-sweep-v2.tar.gz`. SHA256: `d0f85b8e0372534271f9a1dbb51bff13c333cce14202ec683ed7d2e8e1804e12`. The legacy archive name is preserved for provenance.
