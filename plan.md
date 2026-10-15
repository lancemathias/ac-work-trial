# Artifact preparation and repo cleanup plan

## Goals
* Deliver all deliverables required in the task brief in a clean, auditable manner
* Make sure all work is tracked
* Ensure no loose ends remain, code is easy to follow, and all changes are explained
## Artifacts to prepare
* Presentation: slides or document cataloging all experiments we did, datasets and architectures explored, and justification for each
* Research document: longer document explaining methodology, pinned versions for models and data, and repro steps
* Codebase: existing github repo should be cleaned up so that main branch tracks the final deliverable model, we can use branches for experiments but these should be logged in research document. additionally, commits marking major changes should be marked in research doc.
* Wandb workspace: authoritative store of what was run, its corresponding configs, and a note on the experiment.
* Artifacts: pinned checkpoints, datasets, and important eval results should be stored in Modal and referenced in the research doc
## Repo cleanup
* see above. any commits or branches not mentioned in the research document (add a section about codebase map) is assumed to be intermediate work.
* work marked as autoresearch can be lumped together and held to a lower standard of justification, but must have the same repro standard. all autoresearch work should be explicitly marked as such.
* avoid rewriting github history to the maximum extent possible
## Order of work
* make sure ALL work in github is committed and pushed, use branches and only commit to main (via a PR) if the run represents the best candidate so far
* ensure all important artifacts are uploaded to modal
* write a NEW research document. The format should be like a narrative, with headings representing action taken/experiment run following chronological order. The first entry should be "background research" and summarize our findings.
## Open questions
* runs are still ongoing. do not disturb new runs.


## Accepted checkpoint policy

Reference only the accepted checkpoint for each run in the report and run index.
Other saved checkpoints remain reachable through that run's artifact directory.

## Execution record

- Experiment and preservation branches pushed over SSH; main unchanged.
- Thirty completed scored conditions imported into W&B as retrospective evidence.
- Unique uncommitted dataset text preserved in content-addressed Modal paths.
- Accepted head checkpoints remotely hashed against original receipts.
- New chronological report, presentation document, inference package and reproduction instructions prepared on the handoff branch.
- Active experiments remain untouched. The selected completed candidate is a snapshot, not a claim that ongoing work has finished.
