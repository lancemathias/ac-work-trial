# Modal work-trial environment

For the Applied Compute work trial in `/Users/bwt/ac-worktrial`, use Modal
workspace `ac-worktrials`, profile `ac-worktrial`, and environment
`worktrial-lance` for every future run, deployment, volume, and secret lookup.
The profile name is not the environment name. Do not create or use a Modal
environment named `ac-worktrial`. This is the user's instruction of 2026-10-14
and applies across sessions. Keep research code and durable results outside
Modal: version control code, configs, seeds, traces, metrics and documents;
export large tensor artifacts locally with checksums before cleaning Modal.
Do not revert research code or results when cleaning up Modal resources.
