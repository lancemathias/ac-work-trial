# E1 cached-feature replication

| Measure | E1 live backbone | E1 cached features |
|---|---:|---:|
| Training + dev evaluations | 65.90 min | 6.72 min |
| Optimizer steps | 282 | 212 |
| Selected epoch | 2 | 1 |
| Calibrated dev accuracy | 66.93% | 66.73% |
| Calibrated dev NLL | 0.8645 | 0.8507 |
| Calibrated dev Brier | 0.4526 | 0.4409 |
| Calibrated dev ECE-15 | 0.0722 | 0.0378 |
| Fitted temperature | 0.79510347 | 0.96732497 |

For the same 212 optimizer windows (13,467 decisions), total window time fell from 2527.1s to 314.8s: **8.03× speedup**. Median paired throughput speedup: 7.51×.

Cache verification and GPU loading cost **3.10 minutes**, excluded from training timing. Cache extraction itself is a pre-existing one-off cost and is not included; this is not a claim about complete cache-creation-to-checkpoint wall time.

Same configuration, initialization and minibatches; CUDA training trajectories and selected epochs differ. Do not attribute metric differences to a quality benefit of caching.

No second benchmark/test score is claimed here; these are the original independent development and calibration partitions.

GPU trained-head diagnostic: on four probes, cached/live features, logits and losses were bit-identical. Maximum cached/live gradient difference was 1.91e-6; repeated live gradients also varied (up to 9.54e-7). These probes support cache correctness, not bitwise reproducibility of the entire optimization trajectory.
