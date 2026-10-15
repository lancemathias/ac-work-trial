# Autoresearch experiment journal

Primary metric: fixed 11-suite calibrated coverage-adjusted macro accuracy, averaging BBH subtasks before the suite macro. Panel repeatedly observed; no claim of an untouched holdout.

| Experiment | Hypothesis/change | Fixed composite | Status |
|---|---|---:|---|
| Jev 1.13.0 | Existing matched API baseline | 85.82% | Complete |
| E1 | Original 9B head training | 57.35% | Complete |
| E2 | Revised v2 data and recipe | 60.19% | Complete |
| E3 | Medium head, expanded v3 data | 59.85% | Complete; -0.34 pp versus E2, single seed |
| E4 | E3 recipe with frozen Qwen3.8-27B and fresh cached features | Pending | Extracting features; 17 CPU tests passed |
| E5 | E4 recipe, bidirectional full attention | Pending | Queued after E4; 19 CPU tests passed |

Validation comparisons: E2 original 73.66%, E2 Medium 73.28%, E2 Small 68.70%. E3 validation population differs; do not compare aggregates as matched estimates. Medium versus original changes optimizer jointly.

## E3 interpretation

Calibrated composite 59.8528%, versus E2 60.1925% (-0.3397 percentage points). E3 is not a demonstrated improvement. Single seed; no significance claim. Dev accuracy 73.6712%, NLL 0.607045, temperature 1.106902. Validation populations differ from E2. Training and test artifacts exported locally with checksum verification under `outputs/model-artifacts/`; lightweight receipts are versioned under `outputs/training/` and `outputs/test/`.

Persistent dominance: RAGTruth False 200/200 (gold False 132/200); TruthfulQA False 198/200 (gold False 127/200). Small BBH subtasks also have one-label outputs; treat those as diagnostic flags with small sample counts. Source audit finds 6,928 Choice and 1,000 Noul records in v3; all Noul comes from MultiRC, with 454 True / 546 False overall. E3 Noul dev accuracy is 50% on 94 records, versus Choice 76.78% on 715. Thus absence of Noul training or simple all-false label imbalance is not established. Broader binary entailment/evidence data is a candidate hypothesis, to be chosen after the controlled backbone/mask tests.

## Paired regression diagnostics and E4 interruption (2026-10-15 UTC)

Checksum-verified E1/E2/E3 traces reproduce the project composite. E3 minus E1 is +2.498 percentage points (paired native-group bootstrap 95% interval +0.237 to +4.724); E3 minus E2 is -0.340 points (interval -2.090 to +1.403). These retrospective, repeatedly observed panel intervals exclude seed variability, multiple comparisons and selection uncertainty.

Class recall supports the requested mechanisms: E3 TabFact False recall falls to 6.67% from E1's 50.48%; Circa Yes recall falls from 76.14% to zero while Probably-yes predictions dominate; ContractNLI Contradiction recall falls from 8.33% to zero. E1 was already poor on Contradiction. RAGTruth remains all-False in both. These associations do not establish which source removal caused the changes. Full counts and confusion matrices are in `outputs/autoresearch/e1-class-recall.json` and `e3-class-recall.json`.

E4's CPU orchestrator was preempted during feature extraction; automatic replay failed because its run directory existed. The original app stopped with zero tasks. Durable cache progress held 7,232/7,928 rows in 113 shards. One manual extraction-only recovery was submitted with original run ID and unchanged experimental conditions, exact cache identity and shard hash checks, and refusal if training/split/test directories already exist. Existing failure evidence and data are preserved. Recovery app: `ap-3QUTaA36fROgO4F9Hj9HjO`; source commit `7597d8d`. E5 and source grid retain their existing sequential dependencies.


## Deferred readout CPU feasibility (2026-10-15 UTC)

Actual pinned Qwen3.8 tokenizer exposed two implementation constraints before GPU evaluation: chat-template output is a BatchEncoding, and numeric codes above nine are not single tokens. The former is normalized to a checked flat integer list; the latter is replaced with fixed A..Z option codes, preregistered before predictions. The numeric preflight failure receipt is preserved. Letter-code preflight covers 809 dev, 857 calibration and all 2,111 test requests, with distinct single-token option codes and no truncation (maximum test length 8,169 tokens). All 28 pinned-image CPU correctness tests pass, including selected-row/full-LM-logit parity and the mapping-return regression. This establishes implementation feasibility only; no pretrained readout performance is measured. GPU candidate stays deferred behind requested source add-backs and polarity diagnostic. Local test attempts lacked Torch, so authoritative tests ran in the pinned Modal CPU image.


## E4 physical-device resume correction (2026-10-15 06:01 UTC)

Live container inspection confirmed the first manual recovery used a different A100 UUID with the same device, driver 610.57.04 and memory size. The legacy metadata hash included that UUID, guaranteeing its resume identity comparison would fail. The app was explicitly stopped before completing setup; zero containers were confirmed before replacement. No training outcomes were produced.

Source `598fb32` separates only this physical identifier from strict computational identity, preserves original metadata, stores replacement metadata separately, and requires bitwise equality of input IDs and hidden states for both the first and last cached four-record batches before any shard reuse. Every reused shard still requires its SHA256. Driver, dtype, model, seed, attention, data, code and other backend fields remain strict. All 20 pinned-image CPU tests pass, including rejection of altered computation settings, tampered metadata and changed replay tensors. Corrected recovery app `ap-rwv4aBrzqpvzj9FU5VcQSZ` is submitted; actual full-model replay gate remains unverified. This is a diagnosed correction, not a blind retry.


E4 legacy resume live gate subsequently **passed**: eight records from the first and last cached four-record batches have bitwise-equal token IDs and final BF16 hidden states on the replacement A100. Exact metadata migration is limited to physical UUID and the absent legacy native-mode annotations; all other computational identity fields match. Receipt exported with checksum to `outputs/autoresearch/e4-live-resume-parity.json`. Extraction continues from 7,232 records; no training/test outcome yet.
