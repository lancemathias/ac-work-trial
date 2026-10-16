# Source add-back configs

The seven condition JSON files are the training configs. All explicitly use
`selection_metric: accuracy`, with the earliest epoch winning ties and a separate
calibration-only temperature fit. Each condition has 6,512 train decisions,
809 dev decisions, 857 calibration decisions and 510 optimizer updates.

`union-cache-source.json` is a frozen feature-extraction provenance snapshot. Its
nested historical E3 purpose text mentions NLL selection, but extraction fits no
head and selects no checkpoint. Preserve this captured snapshot while its cache
is being built; changing its metadata would change cache identity. It does not
override any condition's accuracy selection.

The first-round grid measures source additions against the equal-sized fresh
existing-source control in the same optimized runtime. Historical E3 differs in
runtime and selection history, so its test score is a secondary reference rather
than the matched causal control.
