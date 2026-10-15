"""Next resident readout service; CPU-check only until current GPU work retires."""
import json,sys,time,subprocess,uuid
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,DEADLINE,image as base_image,models,results,secret
APP='ac-worktrial-readout-candidates-warm';IDLE=1200
image=base_image
if modal.is_local():
 for name in ['fewshot-quality-v2-preparation.json','fewshot-quality-v2-runtime-cpu.json','selective-cached-runtime-cpu.json']:
  image=image.add_local_file(ROOT/'outputs/autoresearch'/name,'/project/'+name)
 image=image.add_local_file(ROOT/'data/training/fewshot-seven-quality-v2/demonstrations.json','/project/fewshot-quality-v2/demonstrations.json')
app=modal.App(APP)

def validate():
 from training_data import digest
 proofs={}
 for name in ['fewshot-quality-v2-preparation.json','fewshot-quality-v2-runtime-cpu.json','selective-cached-runtime-cpu.json']:
  proof=json.loads((Path('/project')/name).read_text())
  if proof['status']!='passed':raise RuntimeError('Passed CPU proof required: '+name)
  for source,expected in proof['source_sha256'].items():
   if digest(ROOT/source)!=expected:raise ValueError('CPU-tested source changed: '+source)
  proofs[name]=digest(Path('/project')/name)
 return proofs

def window(seconds=0):
 if time.time()+seconds+IDLE+60>=DEADLINE:raise TimeoutError('Reservation includes idle expiry and worker grace')

@app.cls(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=12*3600,retries=0,min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=IDLE,volumes={'/models':models.read_only(),'/results':results},secrets=[secret])
class ReadoutCandidates:
 @modal.enter()
 def setup(self):
  from resident_backbone import BackbonePool
  from training_data import digest
  validate();window();self.pool=BackbonePool();self.failed=False;self.worker_id=uuid.uuid4().hex;self.source_sha256=digest(__file__);self.capacity=None
  self.gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
 @modal.method()
 def ping(self):
  window();return dict(worker_id=self.worker_id,source_sha256=self.source_sha256,gpu=self.gpu,failed=self.failed,model_loads=self.pool.loads,model_hits=self.pool.hits)
 @modal.method()
 def fewshot_smoke(self,run_id):
  from training_data import validate_run_id
  from fewshot_smoke import smoke
  validate_run_id(run_id);window(3000)
  if self.failed:raise RuntimeError('Diagnose previous worker failure before reuse')
  try:
   with self.pool.activate():value=smoke(Path('/results')/run_id,DEADLINE,results.commit)
   self.capacity=value
   return dict(value,worker_id=self.worker_id,gpu=self.gpu,model_loads=self.pool.loads,model_hits=self.pool.hits)
  except Exception:self.failed=True;raise
 @modal.method()
 def fewshot_trial(self,run_id,commit):
  from training_data import validate_run_id
  from thinking_staged_budget import gate
  from evaluate_fewshot import evaluate
  validate_run_id(run_id)
  if self.failed:raise RuntimeError('Diagnose previous worker failure before reuse')
  if self.capacity is None:raise RuntimeError('Same resident worker must first pass fewshot capacity smoke')
  budget=gate(self.capacity,'development',time.time(),DEADLINE-60)
  if budget['status']!='passed':return dict(status='held_for_development_budget',development_budget=budget,worker_id=self.worker_id)
  window(budget['estimated_trial_seconds']+600)
  try:
   with self.pool.activate():value=evaluate(Path('/results')/run_id,commit,DEADLINE,results.commit,before_test=lambda:gate(self.capacity,'test',time.time(),DEADLINE-60))
   return dict(value,development_budget=budget,worker_id=self.worker_id,gpu=self.gpu,model_loads=self.pool.loads,model_hits=self.pool.hits)
  except Exception:self.failed=True;raise

@app.function(image=image,cpu=2,memory=8192,timeout=600,retries=0,scaledown_window=2)
def check_imports():
 from evaluate_fewshot import evaluate
 from fewshot_smoke import smoke
 from training_data import digest
 proofs=validate()
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_fewshot_capacity','test_fewshot_runtime'],cwd='/project/tests',capture_output=True,text=True)
 return dict(status='passed' if p.returncode==0 else 'failed',tests=p.stderr,proof_sha256=proofs,worker_source_sha256=digest(__file__),GPU_allocated=False,entrypoints=['ReadoutCandidates.fewshot_smoke','ReadoutCandidates.fewshot_trial'],qualification='CPU imports, source guards and5 mock/capacity contracts; no real GPU capacity or inference.')
@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/readout-candidates-worker-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior worker validation')
 value=check_imports.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
 if value['status']!='passed':raise RuntimeError('Candidate resident worker CPU check failed')
