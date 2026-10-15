"""Resident27B order-pooling trial. Launch only after the source worker is released."""
import json,time,subprocess,sys,uuid
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,DEADLINE,image as base_image,models,results,secret
APP_NAME='ac-worktrial-readout-order-warm';IDLE=1200
image=base_image
if modal.is_local():
 image=(image.add_local_file(ROOT/'outputs/autoresearch/readout-order-preparation.json','/project/readout-order-preflight.json')
  .add_local_file(ROOT/'outputs/autoresearch/readout-order-runtime-cpu.json','/project/readout-order-runtime-cpu.json')
  .add_local_file(ROOT/'outputs/autoresearch/bounded-thinking-cpu.json','/project/bounded-thinking-cpu.json')
  .add_local_file(ROOT/'outputs/autoresearch/thinking-smoke-capacity-cpu.json','/project/thinking-runtime-cpu.json'))
app=modal.App(APP_NAME)
def require_window(seconds):
 if time.time()+seconds+IDLE+60>DEADLINE:raise TimeoutError('Insufficient reservation including idle expiry')
def validate_runtime():
 from training_data import digest
 proof=json.loads(Path('/project/readout-order-runtime-cpu.json').read_text())
 if proof['status']!='passed':raise RuntimeError('Runtime integration tests required')
 for name,expected in proof['source_sha256'].items():
  if digest(Path('/project')/name)!=expected:raise ValueError('Tested runtime source changed: '+name)
 return proof
@app.cls(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=12*3600,retries=0,min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=IDLE,volumes={'/models':models.read_only(),'/results':results},secrets=[secret])
class ReadoutWorker:
 @modal.enter()
 def setup(self):
  from resident_backbone import BackbonePool
  validate_runtime();self.pool=BackbonePool();self.failed=False;self.worker_id=uuid.uuid4().hex
  self.gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
 @modal.method()
 def ping(self):
  require_window(0);return {'worker_id':self.worker_id,'gpu':self.gpu,'failed':self.failed,'model_loads':self.pool.loads,'model_hits':self.pool.hits}
 @modal.method()
 def trial(self,run_id,commit):
  from training_data import validate_run_id
  from evaluate_readout_order import evaluate
  validate_run_id(run_id);require_window(3600)
  if self.failed:raise RuntimeError('Previous failure requires diagnosis')
  try:
   with self.pool.activate():value=evaluate(Path('/results')/run_id,commit,results.commit)
   return dict(value,worker_id=self.worker_id,gpu=self.gpu,model_loads=self.pool.loads,model_hits=self.pool.hits)
  except Exception:self.failed=True;raise
 @modal.method()
 def thinking_smoke(self,run_id):
  from training_data import validate_run_id
  from bounded_thinking_smoke import smoke
  validate_run_id(run_id);require_window(2400);validate_thinking_runtime()
  if self.failed:raise RuntimeError('Previous failure requires diagnosis')
  try:
   with self.pool.activate():value=smoke(Path('/results')/run_id,DEADLINE,results.commit)
   self.thinking_capacity=value
   return dict(value,worker_id=self.worker_id,gpu=self.gpu,model_loads=self.pool.loads,model_hits=self.pool.hits)
  except Exception:self.failed=True;raise
 @modal.method()
 def thinking_trial(self,run_id,commit):
  from training_data import validate_run_id
  from evaluate_bounded_thinking import evaluate
  validate_run_id(run_id);validate_thinking_runtime()
  if self.failed:raise RuntimeError('Previous failure requires diagnosis')
  capacity=getattr(self,'thinking_capacity',None)
  if not capacity or not capacity.get('capacity_validated') or capacity['capacity_gate']['status']!='passed':raise RuntimeError('Same resident worker must first pass bounded thinking capacity smoke')
  require_window(capacity['capacity_gate']['estimated_trial_seconds']+600)
  try:
   with self.pool.activate():value=evaluate(Path('/results')/run_id,commit,DEADLINE,results.commit)
   return dict(value,worker_id=self.worker_id,gpu=self.gpu,model_loads=self.pool.loads,model_hits=self.pool.hits)
  except Exception:self.failed=True;raise

def validate_thinking_runtime():
 from training_data import digest
 proof=json.loads(Path('/project/thinking-runtime-cpu.json').read_text())
 if proof['status']!='passed':raise RuntimeError('Thinking runtime CPU checks required')
 for name,expected in proof['source_sha256'].items():
  if digest(Path('/project')/name)!=expected:raise ValueError('CPU-tested thinking source changed: '+name)
 return proof

@app.function(image=image,cpu=2,memory=4096,timeout=120,retries=0,scaledown_window=2)
def check_imports():
 from evaluate_readout_order import evaluate
 proof=validate_runtime();thinking=validate_thinking_runtime()
 return {'status':'passed','runtime_source_sha256':proof['source_sha256'],'thinking_source_sha256':thinking['source_sha256'],'entrypoints':['ReadoutWorker.trial','ReadoutWorker.thinking_smoke','ReadoutWorker.thinking_trial'],'gpu_allocated':False}
@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 value=check_imports.remote();(ROOT/'outputs/autoresearch/readout-order-worker-cpu.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
@app.local_entrypoint()
def main(run_id:str='readout-native-reverse-pool-v1'):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 receipt=json.loads((ROOT/'outputs/autoresearch/readout-order-worker-cpu.json').read_text())
 if receipt['status']!='passed':raise RuntimeError('CPU import/source validation required')
 containers=json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal','container','list','--profile','ac-worktrial','--env',ENV,'--json'],text=True))
 blocked={'ac-worktrial-math-warm-research','ac-worktrial-warm-research','ac-worktrial-pretrained-readout','ac-worktrial-head-determinism-probe','ac-worktrial-autoresearch-attention-repeats'}
 if any(c['app_name'] in blocked for c in containers):raise RuntimeError('Release prior GPU worker/driver before single-A100 handoff')
 worker=modal.Cls.from_name(APP_NAME,'ReadoutWorker',environment_name=ENV)();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 value=worker.trial.remote(run_id,commit);(ROOT/'outputs/autoresearch/readout-order-result.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2),flush=True)
 while time.time()+IDLE+60<DEADLINE:
  time.sleep(min(300,max(0,DEADLINE-IDLE-60-time.time())))
  if time.time()+IDLE+60<DEADLINE:
   state=worker.ping.remote()
   if state['failed']:raise RuntimeError('Stop keepalive after worker failure')
