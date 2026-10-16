# Bounded-thinking execution

Historical working record. For the current handoff narrative, accepted checkpoints and verified outcomes, see [the research report](research-report.md). Planned/pending statements below describe their recorded stage.


Hypothesis: a fixed128-token native reasoning pass improves semantic letter readout over the original frozen27B no-thinking baseline. Keep native causal attention, frozen weights, code mapping and original input content. Calibrate one positive global temperature using only the original857 nonbenchmark calibration rows. Advance to test only if original809 nonbenchmark dev accuracy improves by at least1 percentage point over82.447466%. Original exploratory test criterion is at least+1pp over77.135889%; the current79.4905% pooled agent remains champion unless a candidate's verified absolute score is higher.

The evaluator's CPU integration tests exercise dev rejection, normal advancement, and a passed dev gate followed by insufficient test budget. Both rejection paths avoid loading/scoring the benchmark panel. All13 generation, prompt-isolation, reservation and evaluator tests pass in the pinned CPU environment. Earlier fixture failure and original9-test proof remain archived. No thinking accuracy has been measured yet.

The real27B capacity smoke passed:8 label-free dev length quantiles plus4 synthetic prompts extended to8256tokens, followed by at most128 generated tokens and full final readout. Peak allocated memory64.18GB. Initial measured batch times24.28,18.68,25.73s led to a14.0h upfront conservative full-run bound, with12.9h remaining. The worker correctly declined full evaluation under that initial policy.

Before any thinking dev/test measurement, a scheduling-only amendment was preregistered. Budget857cal+809dev first using418 batches, including each role's final partial batch. If the unchanged devaccuracy gate passes, recheck current remaining reservation against the entire2111-example test phase (528batches) before loading test rows. A failed second budget check seals the dev/cal result as `held_for_test_budget`; it produces no test inference, score or promotion. Retain the same2x measured slowest-batch factor,600s margin,1200s idle expiry and60s worker grace. Check the hard deadline before every inference batch.

On the updated worker, run two identical12-record label-blind smokes: retain cold timing, then use warm timing for scheduling after the one-time warmup cost is already paid. Both use the exact original examples, synthetic capacity inputs and token limits; no accuracy is computed or used for timing selection. Model and numerical inference helper remain unchanged. Full actual trial remains pending.

The updated code needs a controlled service revision. Retire the original host keepalive and confirm its container is gone before starting the new worker; volumes, exported logits and all durable results stay intact. The new host driver saves durable call IDs before waiting, pins the worker source SHA and rejects worker changes. It retains the loaded backbone after a completed/rejected/held trial for further authorized research. A host failure between spawn and handle persistence still needs manual reconciliation; submission is not transactional.

Evidence: `thinking-staged-budget-amendment.json`, `thinking-staged-runtime-cpu.json`, `readout-order-worker-cpu.json`, and the complete checksum-verified local smoke export. No benchmark task enters training, calibration or dev selection.
