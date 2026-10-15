# Deliver frozen-Qwen native/reverse inference and auditable trial evidence

Main currently presents early training work without a clear accepted inference
entrypoint. This change adds `decision_model.FrozenDecider` for the strongest
verified completed recipe: original frozen Qwen3.8-27B, native/reversed option
orders, semantic probability pooling and the saved calibration temperature.

The observed 11-suite sampled-panel composite is 79.49%. The separate strongest
trained-head condition, E5, is 68.41%; its accepted checkpoint remains indexed.
These are exploratory results on a reused panel, not a claim to beat Jev or to
reproduce vendor full-benchmark scores. The native recipe supports 2–26 options,
text only, and the pinned CUDA numerical runtime.

The handoff adds a chronological research report, presentation document,
reproduction instructions, all completed result comparisons and an accepted-only
checkpoint catalog. Other saved checkpoints remain reachable in their original
Modal run directories. Full experimental runners and preservation snapshots stay
on documented branches; no Git history was rewritten and no active jobs changed.

Validation:
- Research checkout: 174 CPU tests passed.
- Separate delivery checkout: 34 CPU tests passed.
- Restored 125 compact trace files and verified all 30 scored conditions.
- Delivery pooling reproduces every valid archived pooled distribution to 12 decimal places.
- 21 accepted checkpoints remotely hashed against saved receipts.
- No new full-model GPU inference was performed for packaging; measured GPU results retain their original execution provenance.

W&B is the authoritative tracker. Thirty historical benchmark conditions were
imported with explicit retrospective labels and source hashes. Artifact paths,
accepted checkpoints, model/data revisions and major commits are documented.
