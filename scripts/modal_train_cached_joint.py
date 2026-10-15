"""E1 replication: identical training, GPU-resident cached frozen features."""
import json,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENVIRONMENT='worktrial-lance'
if modal.is_local():
    from modal.config import config,_profile
    if _profile!='ac-worktrial' or config.get('environment')!=ENVIRONMENT:
        raise ValueError('Use --profile ac-worktrial --env worktrial-lance')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(ROOT/'requirements-training.txt')
    .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs'))
app=modal.App('ac-worktrial-e1-cached-training')
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENVIRONMENT)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENVIRONMENT)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENVIRONMENT)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENVIRONMENT,required_keys=['WANDB_API_KEY'])

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=12*3600,max_containers=1,
    volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def train(run_id,manifest_hash,commit):
    import sys,time
    sys.path.insert(0,'/project/scripts')
    from train_joint import run
    from training_data import validate_run_id,digest
    validate_run_id(run_id)
    original=Path('/results/e1-qwen35-9b-seed20261014-6f89ea4')
    old=json.loads((original/'config.json').read_text())
    cfg=json.loads(Path('/project/configs/e1-qwen35-9b.json').read_text())
    if cfg!=old['config']:raise ValueError('E1 training config changed')
    manifest=Path('/data/e1-human-approved-20261014/manifest.json')
    if digest(manifest)!=old['manifest_sha256']:raise ValueError('E1 split manifest changed')
    cache=Path('/data/hidden-cache/nulun3tv-qwen35-9b-3b3d8db82da7')
    original_hashes=json.loads((original/'checksums.json').read_text())
    initial_hash=original_hashes['initial-head.safetensors']
    if digest(original/'initial-head.safetensors')!=initial_hash:raise ValueError('E1 initialization evidence changed')
    wall_start=time.monotonic()
    try:
        result=run('/project/configs/e1-qwen35-9b.json',manifest,Path('/results')/run_id,
            checkpoint_callback=results.commit,
            feature_cache=dict(directory=str(cache),expected_manifest_sha256=manifest_hash,
                expected_adapter_sha256=old['code_sha256']['scripts/training_data.py']),
            expected_initial_head_sha256=initial_hash)
        print(json.dumps({'event':'cached_run_complete','wall_seconds':time.monotonic()-wall_start,
            'training_seconds':result['elapsed_seconds'],'commit':commit}),flush=True)
        return result
    finally:results.commit();models.commit()

@app.local_entrypoint()
def main(run_id:str,manifest_hash:str):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    print(json.dumps(train.remote(run_id,manifest_hash,commit),indent=2))
