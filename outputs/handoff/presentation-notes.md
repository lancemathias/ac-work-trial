# Presentation notes

Approximately 30 minutes; notes are kept separately from the slides.

## Frozen-backbone decision models

Goal: explain a principled investigation, not simply maximize a repeatedly observed score. Suggested presentation length: 30 minutes.

## Background research

Explain PPLX’s 1.31M-parameter readout and Clef’s richer evidence routing. The original joint head is 128.06M on 27B. Gemma and other proposed backbones were not run.

## Evaluation contract

Name the preprocessing choices: financial all-agree, fine-grained Circa, GPT-4o JudgeBench cohort, English Belebele, binary MC2 TruthfulQA. These are not verified vendor subsets.

## Completed recipes

Historical comparisons change multiple factors. Best completed inference is 79.49%; best completed trained head is 68.41%. Do not claim cross-seed robustness or a Jev win.

## Original joint head

Describe state/question/option spans and the final-token global vector. The 27B random head has 128,056,324 parameters; 9B has 121,762,820. No independent confidence network.

## E1: learning and collapse

Dev accuracy rises 32.49→66.93%. Gold balance does not explain the JudgeBench minority collapse or sentiment positive bias. Schema layout and semantic binding motivated follow-ups.

## E2 and E3

Original/Medium/Small dev scores: 73.66/73.28/68.70%. Architecture and optimizer jointly change in the original-versus-Medium comparison. More data did not establish better transfer.

## E4: larger frozen backbone

Separate E4 fallback 60.93 from E4 optimized 61.26. Cache identity includes actual numerical runtime. Record cold compilation cost and why cached head training does not benefit from faster backbone kernels.

## Autoresearch: partial bidirectionality

Same optimized numerical regime, v3 partitions and recipe. Separate feature extraction is required. The observed +7.15 pp is single-seed and does not establish a universal improvement.

## Different tasks respond differently

Bidirectional full attention improves finance and judging but worsens Circa and RAGTruth. A composite alone hides failure mechanisms. Discuss the native readout’s broader transfer.

## Autoresearch: repeatability

Automatic backward controls reached 58.60 and 60.85 with no source intervention. The short math fixture runs at 0.844× throughput; do not mistake conditional paired intervals for training uncertainty.

## Autoresearch: source additions

Each condition adds 250 native-labelled examples with equal update counts. MASSIVE passes screening but needs new-seed confirmation. RuleTaker semantic repair was a motivated negative result.

## Pretrained answer semantics

No head or backbone updates. Native and pooled prompts/readouts differ from the joint-head recipe, so this is not an isolated causal estimate of head training. Pooled T=0.9887233144498362.

## Pending and unscored work

W&B finished status is not proof of a completed test. Do not turn partial logits into a score. Active jobs were not changed for cleanup.

## Reproduction and artifacts

Show outputs/handoff/run-index.json and the accepted-model manifest. Distinguish a protocol digest from a weight-file checksum. Point to reproduction commands and explicit model/data pins.

## What we learned

Conclude with limitations: reused exploratory panel, mostly single seed, no vendor-subset reproduction, no Jev superiority. Full experiment catalog and all negative/prepared results are in the report and ledger.
