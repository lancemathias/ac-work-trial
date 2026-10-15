"""Locked E1 benchmark evaluation; one A100 in worktrial-lance."""
from pathlib import Path
import subprocess
import modal
ROOT=Path(__file__).resolve().parents[1]
ENVIRONMENT='worktrial-lance'
from modal.config import config, _profile
if modal.is_local() and (_profile!='ac-worktrial' or config.get('environment')!=ENVIRONMENT):
    raise ValueError('Use --profile ac-worktrial --env worktrial-lance')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(ROOT/'requirements-training.txt')
    .add_local_dir(ROOT/'scripts','/project/scripts')
    .add_local_dir(ROOT/'data/manifests','/project/data/manifests')
    .add_local_dir(ROOT/'data/processed','/project/data/processed'))
app=modal.App('ac-worktrial-e1-test')
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENVIRONMENT)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENVIRONMENT)

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=3600,max_containers=1,
    volumes={'/models':models,'/results':results})
def run(training_run,evaluation_run,commit):
    import sys
    sys.path.insert(0,'/project/scripts')
    from evaluate_trained_joint import evaluate
    from training_data import validate_run_id
    validate_run_id(training_run);validate_run_id(evaluation_run)
    try:return evaluate(Path('/results')/training_run,Path('/results')/evaluation_run,commit,results.commit)
    finally:results.commit();models.commit()

@app.local_entrypoint()
def main(training_run:str,evaluation_run:str):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from training_data import validate_run_id
    validate_run_id(training_run);validate_run_id(evaluation_run)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run.remote(training_run,evaluation_run,commit)
