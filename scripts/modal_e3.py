"""Queue E3 behind v3 cache assembly, then train and evaluate on one A100."""
import json,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs')
 .add_local_dir(ROOT/'data/manifests','/project/data/manifests').add_local_dir(ROOT/'data/processed','/project/data/processed'))
app=modal.App('ac-worktrial-e3-medium-v3-train-and-benchmark')
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENV)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
CACHE='/data/hidden-cache/qwen35-9b-v3-batch4-ef15c46f4919'

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=12*3600,max_containers=1,scaledown_window=2,
 volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def train_and_benchmark(run_id,commit):
 import gc,sys,torch
 sys.path.insert(0,'/project/scripts')
 from training_data import digest
 from train_joint import run,write_json
 from prepare_e3_splits import prepare
 config=Path('/project/configs/e3-medium-v3.json');cfg=json.loads(config.read_text())
 cache=Path(CACHE);manifest=json.loads((cache/'manifest.json').read_text());identity=manifest['identity']
 for k,expected in [('dataset_sha256',cfg['dataset']['sha256']),('backbone_id',cfg['backbone_id']),('backbone_revision',cfg['backbone_revision'])]:
  if identity[k]!=expected:raise ValueError('Wrong v3 cached features: '+k)
 splits=Path('/results')/(run_id+'-splits')
 mp=prepare(Path('/data'+cfg['dataset']['path']),splits,cfg)
 write_json(splits/'checksums.json',{p.name:digest(p) for p in splits.iterdir() if p.is_file()})
 results.commit()
 training=Path('/results')/run_id;test=Path('/results')/(run_id+'-test')
 try:
  run(config,mp,training,checkpoint_callback=results.commit,feature_cache=dict(directory=CACHE,
   expected_manifest_sha256=digest(cache/'manifest.json'),expected_adapter_sha256=digest('/project/scripts/benchmark_training_data.py'),
   expected_source_config=identity['training_source_config'],expected_extraction_batch_size=4))
  # Release the cached tensors and training head before loading the benchmark backbone.
  gc.collect();torch.cuda.empty_cache()
  from evaluate_trained_joint import evaluate
  evaluate(training,test,commit,results.commit)
  from report_trained_joint import report
  metrics=report(test,label='E3 Medium v3 calibrated')
  write_json(test/'checksums.json',{p.name:digest(p) for p in test.iterdir() if p.is_file() and p.name!='checksums.json'})
  try:
   import wandb
   wb=json.loads((training/'wandb-run.json').read_text());w=wandb.Api().run('lmathias/ac-worktrial-decisions/'+wb['id'])
   w.summary['test_calibrated']={k:v for k,v in metrics.items() if k!='suites'}
   w.summary['test_calibrated_per_suite']=metrics['suites'];w.summary.update()
  except Exception as exc:write_json(test/'wandb-update-warning.json',{'error':type(exc).__name__})
  write_json(test/'pipeline.json',{'status':'complete','training_run':run_id,'evaluation_run':test.name,'commit':commit})
  write_json(test/'checksums.json',{p.name:digest(p) for p in test.iterdir() if p.is_file() and p.name!='checksums.json'})
  return {'status':'complete','run':run_id,'test':test.name,'macro_accuracy_misses_wrong':metrics['macro_coverage_adjusted_accuracy']}
 finally:results.commit();models.commit()

@app.function(image=image,cpu=.25,memory=512,timeout=12*3600,max_containers=1,volumes={'/data':data})
def queued_pipeline(run_id,commit):
 import time
 for _ in range(660):
  data.reload();p=Path(CACHE)/'manifest.json'
  if p.exists():
   manifest=json.loads(p.read_text())
   if manifest.get('status')=='complete' and manifest.get('cached_head_backward_verified'):
    print('v3 cache ready; E3 training and subsequent benchmark submitted to A100',flush=True)
    return train_and_benchmark.remote(run_id,commit)
  time.sleep(60)
 raise TimeoutError('v3 cache did not finish within eleven hours')

@app.local_entrypoint()
def main(run_id:str):
 from modal.config import config,_profile
 if _profile!='ac-worktrial' or config.get('environment')!=ENV:raise ValueError('Use ac-worktrial profile and worktrial-lance environment')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 print(json.dumps(queued_pipeline.remote(run_id,commit),indent=2))
