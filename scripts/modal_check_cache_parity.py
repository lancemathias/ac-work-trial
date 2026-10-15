from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt').add_local_dir(ROOT/'scripts','/project/scripts'))
app=modal.App('ac-worktrial-cache-trained-head-parity')
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENV)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=1800,max_containers=1,volumes={'/data':data,'/models':models,'/results':results})
def run(training):
    import sys,json
    sys.path.insert(0,'/project/scripts')
    from training_data import validate_run_id
    from check_cached_head_parity import check
    validate_run_id(training)
    return check(Path('/results')/training,'/data/hidden-cache/nulun3tv-qwen35-9b-3b3d8db82da7','/data/e1-human-approved-20261014/manifest.json')
@app.local_entrypoint()
def main(training:str):
    import json
    value=run.remote(training)
    out=ROOT/'outputs/training'/training/'trained-head-cache-parity.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
