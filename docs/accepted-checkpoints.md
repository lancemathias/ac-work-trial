# Accepted checkpoints

Only the accepted head is referenced for each trained run. Other saved states remain reachable in its original Modal run directory. Full paths, SHA-256 values, configurations, W&B links and verification receipts are in the [machine-readable index](../outputs/handoff/run-index.json).

All entries below were read from Modal and matched their saved accepted-checkpoint SHA-256.

| Run | Accepted epoch / step | Selection | SHA-256 prefix |
| --- | --- | --- | --- |
| ab1-fast-accuracy-seed20261014-existing-source-control | 5 / 510 | accuracy | `c3d57393c0e1b8bc…` |
| ab1-fast-accuracy-seed20261014-existing-source-control-repeat | 4 / 408 | accuracy | `1d3d33720e575e73…` |
| ab1-math-accuracy-seed20261014-aegis2 | 5 / 510 | accuracy | `6a398730bdfd7b58…` |
| ab1-math-accuracy-seed20261014-banking77 | 5 / 510 | accuracy | `058a3cc6ebddae20…` |
| ab1-math-accuracy-seed20261014-civilcomments | 5 / 510 | accuracy | `6d76b951314cf44e…` |
| ab1-math-accuracy-seed20261014-control-1 | 3 / 306 | accuracy | `ed121bf01b70cec8…` |
| ab1-math-accuracy-seed20261014-control-2 | 3 / 306 | accuracy | `ed121bf01b70cec8…` |
| ab1-math-accuracy-seed20261014-massive | 5 / 510 | accuracy | `addf9d5ee60eb24e…` |
| ab1-math-accuracy-seed20261014-ruletaker | 5 / 510 | accuracy | `17827cb1689dc856…` |
| ab1-math-accuracy-seed20261014-snli-multinli | 3 / 306 | accuracy | `61c25696aea8c050…` |
| e1-cached-qwen35-9b-seed20261014-057b5b1 | 1 / 71 | nll | `6cf49ac2ef1919bb…` |
| e1-qwen35-9b-seed20261014-6f89ea4 | 2 / 142 | nll | `3e555fcb36d2ee6e…` |
| e2-qwen35-9b-v2-cache-seed20261014-18359ca | 4 / 128 | nll | `cadc8d40f9254141…` |
| e3-medium-v3-seed20261014-e7dd4bb | 3 / 294 | nll | `10e5aa47916417a7…` |
| e4-accuracy-v3-seed20261014 | 5 / 490 | accuracy | `296435fe4ef1deb2…` |
| e4-fast-accuracy-v3-seed20261014 | 5 / 490 | accuracy | `bec6e956cbd834e5…` |
| attention-bidir-qwen38-27b-v3-seed20261014 | 2 / 196 | accuracy | `b0efb3dd0290110a…` |
| ruletaker-cwa-explicit-math-seed20261014 | 4 / 408 | accuracy | `b98e3abedb261010…` |
| e4-qwen38-27b-v3-seed20261014-2d9ef9a | 3 / 294 | nll | `343c1cb63f923758…` |
| e2-small-head-seed20261014-eec3a11 | 2 / 64 | nll | `ab74361a55d94cee…` |
| e2-medium-head-seed20261014-eec3a11 | 2 / 64 | nll | `947a16cf0e2bc915…` |

The native/reverse inference recipe has no trained head. Its accepted artifact is the [pinned backbone/readout/pooling/temperature manifest](../decision_model/accepted-model.json), not a new weight file. Failed/interrupted runs without an accepted head are not assigned a fabricated checkpoint.

Descriptive names are mapped to immutable legacy run IDs and storage paths in [resource aliases](../configs/resource-aliases.json). The bidirectional full-attention checkpoint is unchanged.
