# Decision model work trial

Auditable training research and matched decision-model baselines on the 11 tasks
named in Perplexity's pplx-decider-v1-27b model card. Our independently defined
protocol is not a verified reproduction of Perplexity's unpublished panel.

- [Baseline protocol, assumptions and reproduction](docs/baselines.md)
- [Measured accuracy and calibration tables](outputs/baselines-20261014/results.md)
- [Frozen Qwen3.8 random joint head methodology and three seeds](docs/random-joint-baseline.md)
- [Baseline findings](docs/baseline-results.md)
- [Research log](docs/research-log.md)
- [Design and experiment plan](docs/proof-of-concept.md)
- [Dataset sources, splits and conversion rules](docs/benchmark-preprocessing.md)
- [Raw source links, immutable revisions and SHA-256 receipt](data/manifests/benchmark-preprocessing.json)
- [Frozen evaluation IDs and model revisions](data/manifests/baseline-panel.json)
- [Nimble's exact pinned subset manifests](outputs/nimble-pinned-subsets/index.json)

## Artifact policy

Commit runners, tests, dependency pins, research documents, manifests, run configs,
prediction traces, metrics, completion receipts and result tables. Raw datasets,
processed example text, credentials, model weight downloads and scratch files
remain outside Git. Data is attributable and reproducible from the committed
source URLs/revisions plus raw and processed checksums. Review per-source terms
before redistributing datasets. Prediction artifacts retain IDs, labels and model
responses, without copying input contexts; use the frozen IDs to reconstruct them.

Use Modal profile `ac-worktrial` with environment `worktrial-lance` for all future
runs (`--profile ac-worktrial --env worktrial-lance`). The previous trial-created
`ac-worktrial` environment is being removed after a complete local results export.
Code and durable results belong outside Modal: results are versioned in
`outputs/baselines-20261014`; large initial head tensors are archived locally in
`outputs/modal-archive-20261014` with a committed checksum manifest. Model caches
can be downloaded again from the immutable Hub revisions in the run configs. Never overwrite
a finished run with another protocol or silently combine partial runs.

## Run and verify

See docs/baselines.md for environment setup and commands. The default Modal launch
runs PPLX and CLEF-flash concurrently, each on an A100-80GB with a hard one-hour
timeout. CLEF is launched separately after cost review with `--model-key clef`;
skip it if PPLX reaches its cap. Jev uses the key named JEV_API_KEY in local .env.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_baselines.py report
```

Current status is represented by completion receipts and coverage in the results
table. Pending entries are not measured zeroes or completed benchmarks.
