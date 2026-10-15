"""Independent E2 medium/small head ablations on the exact v2 cached splits."""
import json,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs'))
app=modal.App('ac-worktrial-e2-head-size-ablation')
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENV)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=12*3600,max_containers=1,
 volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def train(variant,run_id,commit):
 import sys
 sys.path.insert(0,'/project/scripts')
 from train_joint import run
 from training_data import digest,validate_run_id
 if variant not in ['medium','small']:raise ValueError('Unknown variant')
 validate_run_id(run_id)
 config=Path(f'/project/configs/e2-{variant}-head.json');cfg=json.loads(config.read_text())
 base=json.loads(Path('/results/e2-qwen35-9b-v2-cache-seed20261014-18359ca/config.json').read_text())['config']
 allowed = ['head','purpose','learning_rate','adamw_betas','adamw_eps','adamw_decay_policy']
 if {k:v for k,v in cfg.items() if k not in allowed}!={k:v for k,v in base.items() if k not in allowed}:raise ValueError('Non-head E2 config changed')
 metadata=json.loads(Path('/project/configs/e2-head-ablation-metadata.json').read_text())
 manifest=Path('/results/e2-qwen35-9b-v2-cache-seed20261014-18359ca-splits/manifest.json')
 if digest(manifest)!=metadata['split_manifest_sha256']:raise ValueError('E2 splits changed')
 expected=json.loads(Path('/project/configs/e2-cache-source.json').read_text())['config']['config']
 out=Path('/results')/run_id
 try:
  result=run(config,manifest,out,checkpoint_callback=results.commit,
   feature_cache=dict(directory='/data/hidden-cache/qwen35-9b-v2-batch4-338a3584e896',expected_manifest_sha256='b461903e3990adc1bfbbf649b7e5c658f617dee3e76be4e61baead4fbce8d293',expected_adapter_sha256=digest('/project/scripts/benchmark_training_data.py'),expected_source_config=expected,expected_extraction_batch_size=4))
  from train_joint import write_json
  write_json(out/'ablation.json',dict(variant=variant,commit=commit,parameters=metadata[variant]['parameters'],reference_run='e2-qwen35-9b-v2-cache-seed20261014-18359ca',changed_config_keys=allowed))
  # Include the added ablation receipt in the durable export inventory.
  write_json(out/'checksums.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='checksums.json'})
  return result
 finally:results.commit();models.commit()
@app.local_entrypoint()
def main(variant:str,run_id:str):
 from modal.config import config,_profile
 if _profile!='ac-worktrial' or config.get('environment')!=ENV:raise ValueError('Use authorized profile and environment')
 if variant not in ['medium','small']:raise ValueError('Unknown variant')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 print(json.dumps(train.remote(variant,run_id,commit),indent=2))
