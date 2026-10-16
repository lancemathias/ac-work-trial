"""Warm A100 math-head confirmation. Source trials require a strict repeat gate."""
import json,time,subprocess,uuid,sys
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,DEADLINE,image as source_image,data,models,results,secret,NAMES
APP_NAME='ac-worktrial-math-warm-research';IDLE=1200
image=source_image
if modal.is_local():image=image.add_local_dir(ROOT/'outputs/autoresearch/source-condition-caches','/project/cache-receipts')
app=modal.App(APP_NAME)
INITIAL='909b3c9ff1f565479d2bd81a1346564af2f5f0a433f2bf869fa9e7cad4ed573e'
BATCH='ab1-math-accuracy-seed20261014';READOUT='readout-fast-qwen38-27b-v1'

def window(seconds):
 if time.time()+seconds+IDLE+60>DEADLINE:raise TimeoutError('Reservation includes warm-worker expiry')

def condition(name,run_id,commit,worker):
 import gc,torch
 from training_data import digest,validate_run_id
 from head_probe_io import prepare_run,write_json,seal
 from math_source_repro import runtime_config
 from train_joint import run
 from evaluate_trained_joint import evaluate
 from report_trained_joint import report
 from report_prediction_dominance import summarize
 validate_run_id(run_id)
 if name not in NAMES:raise ValueError('Unregistered condition')
 window(3600);data.reload();results.reload()
 cfgpath=Path('/project/configs/source-addbacks')/(name+'.json');cfg=json.loads(cfgpath.read_text())
 cache=json.loads(Path('/project/cache-receipts/cache-receipts.json').read_text())[name]
 root=Path('/results')/run_id
 request={'run_id':run_id,'condition':name,'commit':commit,'head_sdpa_backend':'math','source_config_sha256':digest(cfgpath),'cache':cache,'initial_head_sha256':INITIAL}
 complete=prepare_run(root,request);results.commit()
 if complete is not None:return complete
 attempts=root/'attempts';attempts.mkdir(exist_ok=True)
 if len(list(attempts.iterdir()))>=3:raise RuntimeError('Three interrupted attempts; stop for diagnosis')
 attempt=attempts/uuid.uuid4().hex;attempt.mkdir()
 write_json(attempt/'started.json',dict(request=request,worker=worker,time=time.time()));results.commit()
 try:
  runtime=runtime_config(cfg,run_id);runtimepath=attempt/'runtime-config.json';write_json(runtimepath,runtime)
  gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  training=attempt/'training';test=attempt/'test'
  run(runtimepath,Path('/project/source-addbacks')/name/'manifest.json',training,checkpoint_callback=results.commit,feature_cache=dict(directory=cache['directory'],expected_manifest_sha256=cache['manifest_sha256'],expected_adapter_sha256=digest('/project/scripts/benchmark_training_data.py'),expected_source_config=cfg,expected_extraction_batch_size=4),expected_initial_head_sha256=INITIAL)
  summary=json.loads((training/'summary.json').read_text())
  if summary['steps']!=510:raise ValueError('Expected 510 updates')
  gc.collect();torch.cuda.empty_cache();evaluate(training,test,commit,results.commit)
  metrics=report(test,label='Math source '+name+' calibrated')
  write_json(test/'prediction-distributions.json',summarize(test/'calibrated-predictions.jsonl'))
  try:
   import wandb
   wb=json.loads((training/'wandb-run.json').read_text());w=wandb.Api().run('lmathias/ac-worktrial-decisions/'+wb['id']);w.summary['test_calibrated']={k:v for k,v in metrics.items() if k!='suites'};w.summary['test_calibrated_per_suite']=metrics['suites'];w.summary.update()
  except Exception as exc:write_json(test/'wandb-update-warning.json',{'error':type(exc).__name__})
  write_json(test/'checksums.json',{p.name:digest(p) for p in test.iterdir() if p.is_file() and p.name!='checksums.json'})
  value={'status':'complete','request':request,'attempt':str(attempt.relative_to(root)),'worker':worker,'composite':metrics['macro_coverage_adjusted_accuracy'],'steps':summary['steps'],'initial_head_sha256':digest(training/'initial-head.safetensors')}
  write_json(root/'result.json',value);seal(root);results.commit();return value
 finally:results.commit();models.commit()

@app.cls(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3600,retries=0,min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=IDLE,volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
class MathWorker:
 @modal.enter()
 def setup(self):
  import torch
  from resident_backbone import BackbonePool
  from kernel_runtime import configure_optimized_runtime
  torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
  configure_optimized_runtime('both');self.pool=BackbonePool();self.failed=False
  self.identity={'worker_id':uuid.uuid4().hex,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()}
 @modal.method()
 def ping(self):
  window(0);return dict(self.identity,failed=self.failed,model_loads=self.pool.loads,model_hits=self.pool.hits)
 @modal.method()
 def run(self,name,run_id,commit):
  if self.failed:raise RuntimeError('Prior failure requires diagnosis')
  try:
   with self.pool.activate():value=condition(name,run_id,commit,self.identity)
   return dict(value,model_loads=self.pool.loads,model_hits=self.pool.hits)
  except Exception:self.failed=True;raise

@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def cpu_check():
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_math_source_repro','test_head_runtime','test_head_probe_io','test_cached_training','test_checkpoint_selection','test_resident_backbone','test_training','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-14000:])
 return {'status':'passed','tests':p.stderr}

@app.function(image=image,cpu=2,memory=8192,timeout=24*3600,retries=0,max_containers=1,volumes={'/results':results})
def pipeline(commit,driver_id):
 from head_probe_io import prepare_run,write_json,seal
 from math_source_repro import compare
 root=Path('/results')/driver_id;request={'commit':commit,'driver_id':driver_id,'batch':BATCH};done=prepare_run(root,request)
 if done is not None:return done
 def status(phase,**kw):write_json(root/'status.json',dict(phase=phase,time=time.time(),**kw));results.commit()
 worker=modal.Cls.from_name(APP_NAME,'MathWorker',environment_name=ENV)()
 try:
  status('waiting_for_readout')
  while True:
   window(7200);results.reload();path=Path('/results')/READOUT/'result.json'
   if path.exists():
    if json.loads(path.read_text()).get('status')!='complete':raise RuntimeError('Readout did not complete; inspect before GPU handoff')
    break
   time.sleep(20)
  completed=[]
  for number in [1,2]:
   status('full_repeat',replica=number)
   completed.append(worker.run.remote('existing-source-control',BATCH+'-control-'+str(number),commit))
   write_json(root/('control-'+str(number)+'.json'),completed[-1]);results.commit()
  if completed[0]['worker']['worker_id']!=completed[1]['worker']['worker_id']:raise RuntimeError('Full controls require same physical worker; interrupted outputs preserved')
  results.reload();folders=[Path('/results')/r['request']['run_id']/r['attempt'] for r in completed]
  gate=compare(*folders);write_json(root/'repeat-gate.json',gate);results.commit()
  if gate['status']!='passed':raise RuntimeError('Exact full-repeat gate failed; source queue stays held')
  status('repeat_gate_passed_keep_warm',controls=completed)
  while time.time()+IDLE+60<DEADLINE:
   time.sleep(min(300,max(0,DEADLINE-IDLE-60-time.time())))
   if time.time()+IDLE+60<DEADLINE:
    state=worker.ping.remote()
    if state['failed']:raise RuntimeError('Worker failure; stop keepalive')
  status('reservation_expiry');return {'status':'complete','request':request,'controls':completed,'gate':gate}
 except Exception as exc:status('failed',error=type(exc).__name__,message=str(exc));raise

@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 value=cpu_check.remote();(ROOT/'outputs/autoresearch/math-source-cpu.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))

@app.local_entrypoint()
def main(driver_id:str='math-source-repro-driver-v1'):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 receipt=json.loads((ROOT/'outputs/autoresearch/math-source-cpu.json').read_text())
 if receipt['status']!='passed':raise RuntimeError('CPU validation required')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 print(json.dumps(pipeline.remote(commit,driver_id),indent=2))
