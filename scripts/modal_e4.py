"""Sequential E4 setup, training and locked benchmark; waits for E3's A100."""
import json,time,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
DEADLINE=1792125869  # 2026-10-16 04:44:29 UTC
E3='e3-medium-v3-seed20261014-e7dd4bb'
CACHE='/data/hidden-cache/e4-qwen38-27b-v3-batch4-ef15c46f4919'
app=modal.App('ac-worktrial-autoresearch-e4')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs')
 .add_local_dir(ROOT/'tests','/project/tests').add_local_dir(ROOT/'data/manifests','/project/data/manifests').add_local_dir(ROOT/'data/processed','/project/data/processed'))
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENV)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
@app.function(image=image,cpu=4,memory=8192,timeout=600)
def verify_cpu():
 import subprocess,sys
 p=subprocess.run([sys.executable,'-m','unittest','-v','test_cache_resume','test_training','test_cached_training','test_training_runtime','test_adamw_groups','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 result={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 if p.returncode:raise RuntimeError('CPU validation failed: '+p.stderr[-12000:])
 return result
def require_budget(seconds):
 if time.time()+seconds>DEADLINE:raise TimeoutError('Insufficient reserved A100 time; refusing GPU stage')
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3*3600,max_containers=1,scaledown_window=2,
 volumes={'/data':data,'/models':models})
def extract():
 require_budget(3*3600)
 import sys
 sys.path.insert(0,'/project/scripts')
 from cache_hidden_runtime_batch4 import extract as extract_cache
 snapshot=json.loads(Path('/project/configs/e4-cache-source.json').read_text());cfg=snapshot['config']['config']
 try:
  result=extract_cache(snapshot,Path('/data'+cfg['dataset']['path']),'/project/configs/normalized-training.schema.json',Path(CACHE),checkpoint=data.commit)
  return {'status':result['status'],'records':result['identity']['record_count'],'feature_bytes':result['identity']['feature_bytes']}
 finally:data.commit();models.commit()
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3600,max_containers=1,scaledown_window=2,
 volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def train_and_benchmark(run_id,commit):
 require_budget(3600)
 import gc,sys,torch
 sys.path.insert(0,'/project/scripts')
 from training_data import digest
 from train_joint import run,write_json
 from prepare_e3_splits import prepare
 cfg_path=Path('/project/configs/e4-qwen38-27b-v3.json');cfg=json.loads(cfg_path.read_text())
 data.reload();manifest=json.loads((Path(CACHE)/'manifest.json').read_text());identity=manifest['identity']
 for k,expected in [('dataset_sha256',cfg['dataset']['sha256']),('backbone_id',cfg['backbone_id']),('backbone_revision',cfg['backbone_revision'])]:
  if identity[k]!=expected:raise ValueError('Cache mismatch: '+k)
 splits=Path('/results')/(run_id+'-splits');mp=prepare(Path('/data'+cfg['dataset']['path']),splits,cfg)
 audit=json.loads(Path('/project/configs/e4-data-audit.json').read_text())
 if not audit['approved'] or audit['dataset_sha256']!=cfg['dataset']['sha256']:raise ValueError('Missing data exclusion audit')
 write_json(splits/'hygiene-report.json',audit)
 split_manifest=json.loads(mp.read_text());split_manifest['hygiene']['report']['sha256']=digest(splits/'hygiene-report.json');write_json(mp,split_manifest)
 # Exact E3 partition equivalence is mandatory for this backbone comparison.
 e3=Path('/results')/(E3+'-splits')/'manifest.json';e3m=json.loads(e3.read_text());e4m=json.loads(mp.read_text())
 for role in ['train','dev','calibration']:
  if e3m['splits'][role]['points']['sha256']!=e4m['splits'][role]['points']['sha256']:raise ValueError('E3/E4 split differs: '+role)
 write_json(splits/'checksums.json',{p.name:digest(p) for p in splits.iterdir() if p.is_file()});results.commit()
 training=Path('/results')/run_id;test=Path('/results')/(run_id+'-test')
 try:
  run(cfg_path,mp,training,checkpoint_callback=results.commit,feature_cache=dict(directory=CACHE,
   expected_manifest_sha256=digest(Path(CACHE)/'manifest.json'),expected_adapter_sha256=digest('/project/scripts/benchmark_training_data.py'),
   expected_source_config=identity['training_source_config'],expected_extraction_batch_size=4))
  gc.collect();torch.cuda.empty_cache()
  from evaluate_trained_joint import evaluate
  evaluate(training,test,commit,results.commit)
  from report_trained_joint import report
  metrics=report(test,label='E4 frozen Qwen3.8-27B Medium v3 calibrated')
  try:
   import wandb
   wb=json.loads((training/'wandb-run.json').read_text());w=wandb.Api().run('lmathias/ac-worktrial-decisions/'+wb['id'])
   w.summary['test_calibrated']={k:v for k,v in metrics.items() if k!='suites'};w.summary['test_calibrated_per_suite']=metrics['suites'];w.summary.update()
  except Exception as exc:write_json(test/'wandb-update-warning.json',{'error':type(exc).__name__})
  write_json(test/'pipeline.json',{'status':'complete','training_run':run_id,'commit':commit})
  write_json(test/'checksums.json',{p.name:digest(p) for p in test.iterdir() if p.is_file() and p.name!='checksums.json'})
  return {'status':'complete','run_id':run_id,'macro_accuracy_misses_wrong':metrics['macro_coverage_adjusted_accuracy']}
 finally:results.commit();models.commit()
@app.function(image=image,cpu=.25,memory=512,timeout=6*3600,max_containers=1,volumes={'/results':results})
def pipeline(run_id,commit):
 out=Path('/results')/(run_id+'-pipeline');out.mkdir(exist_ok=False)
 def status(phase,**kwargs):
  p=out/'status.json';p.write_text(json.dumps({'phase':phase,'run_id':run_id,'commit':commit,'timestamp':time.time(),**kwargs},indent=2)+'\n');results.commit()
 try:
  status('validating_cpu');validation=verify_cpu.remote()
  (out/'cpu-validation.json').write_text(json.dumps(validation,indent=2)+'\n');results.commit()
  status('waiting_for_e3')
  while True:
   require_budget(4*3600);results.reload()
   if (Path('/results')/(E3+'-test')/'pipeline.json').exists():break
   time.sleep(30)
  status('extracting_frozen_features');setup=extract.remote();status('training_and_benchmark',setup=setup)
  result=train_and_benchmark.remote(run_id,commit);status('complete',result=result);return result
 except Exception as exc:
  status('failed',error=type(exc).__name__,message=str(exc));raise
@app.function(image=image,cpu=.25,memory=512,timeout=6*3600,retries=0,max_containers=1,volumes={'/results':results})
def recover_extraction(run_id,commit):
 # Manual, single recovery after verified stopped app; never replay training.
 import sys
 sys.path.insert(0,'/project/scripts')
 results.reload()
 out=Path('/results')/(run_id+'-pipeline')
 previous=json.loads((out/'status.json').read_text())
 from cache_resume import require_extraction_phase
 prior_marker=out/'manual-recovery.json'
 require_extraction_phase(previous,json.loads(prior_marker.read_text()) if prior_marker.exists() else None)
 if any((Path('/results')/(run_id+suffix)).exists() for suffix in ['', '-test', '-splits']):raise ValueError('Refuse recovery after training began')
 marker=out/'physical-gpu-recovery.json'
 with marker.open('x') as f:json.dump({'previous':previous,'recovery_commit':commit,'timestamp':time.time()},f,indent=2)
 results.commit()
 def status(phase,**kwargs):
  (out/'status.json').write_text(json.dumps({'phase':phase,'run_id':run_id,'commit':commit,'timestamp':time.time(),'manual_recovery':True,**kwargs},indent=2)+'\n');results.commit()
 try:
  validation=verify_cpu.remote()
  (out/'recovery-cpu-validation.json').write_text(json.dumps(validation,indent=2)+'\n');results.commit()
  # Extractor checks exact identity and SHA256 of every reused shard.
  setup=extract.remote();status('training_and_benchmark',setup=setup)
  result=train_and_benchmark.remote(run_id,commit);status('complete',result=result);return result
 except Exception as exc:
  status('failed',error=type(exc).__name__,message=str(exc));raise
@app.local_entrypoint()
def main(run_id:str,recover:bool=False,cpu_only:bool=False):
 from modal.config import config,_profile
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 audit=json.loads((ROOT/'outputs/autoresearch/data-audit.json').read_text())
 assert audit['approved'] and audit['dataset_sha256']==json.loads((ROOT/'configs/e4-qwen38-27b-v3.json').read_text())['dataset']['sha256']
 if cpu_only:
  result=verify_cpu.remote();(ROOT/'outputs/autoresearch/cache-resume-cpu-tests.json').write_text(json.dumps(result,indent=2)+'\n');print(result['stderr'][-3000:]);return
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 print(json.dumps((recover_extraction if recover else pipeline).remote(run_id,commit),indent=2))
