"""Cache only new v4 views with the v3.1 pinned 27B backbone."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1] if modal.is_local() else Path('/project')
ENV='worktrial-lance'
app=modal.App('ac-worktrial-v4-27b-feature-cache')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt').pip_install_from_requirements(ROOT/'requirements-kernels.txt',extra_options='--no-deps').add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs').add_local_dir(ROOT/'data/training/v4-wice-evidence-massive','/project/data/training/v4-wice-evidence-massive'))
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
@app.function(image=image,cpu=4,memory=16384,timeout=1800,volumes={'/models':models.read_only()})
def preflight():
    import sys
    sys.path.insert(0,'/project/scripts')
    from cache_v4_readout import prepare
    return prepare()[2]
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=12*3600,retries=0,max_containers=1,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results})
def cache():
    import sys
    sys.path.insert(0,'/project/scripts')
    from cache_v4_readout import run
    return run('/results',results.commit)
@app.local_entrypoint()
def main():
    from modal.config import _profile,config
    assert _profile=='ac-worktrial' and config.get('environment')==ENV
    proof=preflight.remote()
    target=ROOT/'outputs/data-releases/v4-wice-evidence-massive'
    (target/'tokenizer-preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
    call=cache.spawn()
    receipt={'function_call_id':call.object_id,'profile':'ac-worktrial','workspace':'ac-worktrials','environment':ENV,'preflight':proof}
    (target/'cache-launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
