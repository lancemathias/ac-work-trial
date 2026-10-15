"""Seven matched source conditions, one cache setup, one A100, no blind retries."""
import json,time,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';DEADLINE=1792125869
E5='e5-qwen38-27b-bidir-v3-seed20261014-af01134'
NAMES=['existing-source-control','ruletaker','aegis2','civilcomments','snli-multinli','massive','banking77']
BASE='/data/hidden-cache/qwen35-9b-v3-batch4-ef15c46f4919';UNION='/data/hidden-cache/source-addback-round1-union-v2'
app=modal.App('ac-worktrial-autoresearch-source-addbacks')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs').add_local_dir(ROOT/'tests','/project/tests')
 .add_local_dir(ROOT/'data/manifests','/project/data/manifests').add_local_dir(ROOT/'data/processed','/project/data/processed')
 .add_local_dir(ROOT/'data/training/source-addbacks-v2','/project/source-addbacks')
 .add_local_file(ROOT/'outputs/training/e3-medium-v3-seed20261014-e7dd4bb/config.json','/project/outputs/training/e3-medium-v3-seed20261014-e7dd4bb/config.json'))
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENV);models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV);results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
def budget(seconds):
 if time.time()+seconds>DEADLINE:raise TimeoutError('Reserved A100 window insufficient for this bounded stage')
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def verify_cpu():
 import sys
 sys.path.insert(0,'/project/scripts')
 from verify_source_addbacks import verify
 data_check=verify('/project/source-addbacks')
 p=subprocess.run([sys.executable,'-m','unittest','-v','test_cache_composition','test_binary_views','test_backbone_attention','test_training','test_cached_training','test_training_runtime','test_adamw_groups','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 return {'status':'passed' if p.returncode==0 else 'failed','data':data_check,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=7200,max_containers=1,scaledown_window=2,retries=0,volumes={'/data':data,'/models':models})
def extract_union():
 budget(7200)
 import sys
 sys.path.insert(0,'/project/scripts')
 from cache_hidden_runtime_batch4 import extract
 snapshot=json.loads(Path('/project/configs/source-addbacks/union-cache-source.json').read_text())
 try:
  m=extract(snapshot,Path('/project/source-addbacks/union-additions.jsonl'),'/project/configs/normalized-training.schema.json',Path(UNION),checkpoint=data.commit)
  return {'status':'complete','records':m['identity']['record_count']}
 finally:data.commit();models.commit()
@app.function(image=image,cpu=8,memory=16384,timeout=7200,retries=0,scaledown_window=2,volumes={'/data':data})
def compose_conditions(batch_id):
 import sys
 sys.path.insert(0,'/project/scripts')
 from compose_feature_cache import compose
 from training_data import load_splits,digest
 data.reload();parts=[{'directory':Path(p),'manifest_sha256':digest(Path(p)/'manifest.json')} for p in [BASE,UNION]];receipts={}
 for name in NAMES:
  cfg=json.loads(Path('/project/configs/source-addbacks/'+name+'.json').read_text());splits,_=load_splits('/project/source-addbacks/'+name+'/manifest.json');rows=[r for values in splits.values() for r in values]
  out=Path('/data/hidden-cache')/(batch_id+'-'+name)
  m=compose(parts,out,rows,cfg,allow_source_subset=True);data.commit()
  receipts[name]={'directory':str(out),'manifest_sha256':digest(out/'manifest.json'),'records':m['identity']['record_count']}
  print(json.dumps({'event':'condition_cache_composed','condition':name,'records':m['identity']['record_count']}),flush=True)
 return receipts
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3600,max_containers=1,retries=0,scaledown_window=2,volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def train_and_benchmark(name,run_id,commit,cache):
 budget(3600)
 import gc,sys,torch
 sys.path.insert(0,'/project/scripts')
 from train_joint import run,write_json
 from training_data import digest
 cfgpath=Path('/project/configs/source-addbacks/'+name+'.json');cfg=json.loads(cfgpath.read_text());training=Path('/results')/run_id;test=Path('/results')/(run_id+'-test')
 try:
  data.reload();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  run(cfgpath,Path('/project/source-addbacks')/name/'manifest.json',training,checkpoint_callback=results.commit,feature_cache=dict(directory=cache['directory'],expected_manifest_sha256=cache['manifest_sha256'],expected_adapter_sha256=digest('/project/scripts/benchmark_training_data.py'),expected_source_config=cfg,expected_extraction_batch_size=4))
  summary=json.loads((training/'summary.json').read_text())
  if summary['steps']!=510:raise ValueError('Actual optimizer updates differ from matched 510-step protocol')
  gc.collect();torch.cuda.empty_cache()
  from evaluate_trained_joint import evaluate
  evaluate(training,test,commit,results.commit)
  from report_trained_joint import report
  metrics=report(test,label='Source add-back '+name+' calibrated')
  from report_prediction_dominance import summarize
  write_json(test/'prediction-distributions.json',summarize(test/'calibrated-predictions.jsonl'))
  try:
   import wandb
   wb=json.loads((training/'wandb-run.json').read_text());w=wandb.Api().run('lmathias/ac-worktrial-decisions/'+wb['id']);w.summary['test_calibrated']={k:v for k,v in metrics.items() if k!='suites'};w.summary['test_calibrated_per_suite']=metrics['suites'];w.summary.update()
  except Exception as exc:write_json(test/'wandb-update-warning.json',{'error':type(exc).__name__})
  write_json(test/'pipeline.json',{'status':'complete','condition':name,'run_id':run_id,'commit':commit});write_json(test/'checksums.json',{p.name:digest(p) for p in test.iterdir() if p.is_file() and p.name!='checksums.json'})
  return {'status':'complete','condition':name,'run_id':run_id,'composite':metrics['macro_coverage_adjusted_accuracy'],'optimizer_updates':summary['steps'],'initial_head_sha256':digest(training/'initial-head.safetensors')}
 finally:results.commit();models.commit()
@app.function(image=image,cpu=.25,memory=1024,timeout=12*3600,retries=0,max_containers=1,volumes={'/results':results})
def pipeline(batch_id,commit):
 out=Path('/results')/(batch_id+'-pipeline');out.mkdir(exist_ok=False)
 def status(phase,**kw):
  (out/'status.json').write_text(json.dumps({'phase':phase,'batch_id':batch_id,'commit':commit,'time':time.time(),**kw},indent=2)+'\n');results.commit()
 try:
  status('cpu_validation');validation=verify_cpu.remote();(out/'cpu-validation.json').write_text(json.dumps(validation,indent=2)+'\n');results.commit()
  if validation['status']!='passed':raise RuntimeError('CPU validation failed; GPU work blocked')
  status('waiting_for_e5')
  while True:
   budget(7200);results.reload()
   if (Path('/results')/(E5+'-test')/'pipeline.json').exists():break
   state=Path('/results')/(E5+'-pipeline')/'status.json'
   if state.exists() and json.loads(state.read_text())['phase']=='failed':raise RuntimeError('E5 failed; halt dependency queue for diagnosis')
   time.sleep(30)
  status('extracting_union');setup=extract_union.remote();status('composing_condition_caches',setup=setup);caches=compose_conditions.remote(batch_id)
  (out/'cache-receipts.json').write_text(json.dumps(caches,indent=2)+'\n');results.commit();completed=[]
  for name in NAMES:
   status('training_and_benchmark',current=name,completed=completed)
   result=train_and_benchmark.remote(name,batch_id+'-'+name,commit,caches[name]);completed.append(result);status('condition_complete',completed=completed)
  status('complete',completed=completed);return {'status':'complete','results':completed}
 except Exception as exc:
  status('failed',error=type(exc).__name__,message=str(exc));return {'status':'failed','error':type(exc).__name__,'message':str(exc)}
@app.local_entrypoint()
def main(batch_id:str):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 check=json.loads((ROOT/'outputs/autoresearch/source-addback-data-validation.json').read_text());assert check['status']=='passed'
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();result=pipeline.remote(batch_id,commit);print(json.dumps(result,indent=2))
 if result['status']!='complete':raise RuntimeError('Source add-back pipeline halted: '+str(result))
