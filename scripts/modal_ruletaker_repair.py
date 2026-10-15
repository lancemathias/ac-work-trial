"""Future resident9B worker for the preregistered same-record semantic repair."""
import json,sys,time,uuid,subprocess
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,DEADLINE,image as source_image,data,models,results,secret
APP_NAME='ac-worktrial-ruletaker-repair-warm';IDLE=1200
image=source_image
if modal.is_local():
 image=(image.add_local_dir(ROOT/'data/training/ruletaker-cwa-explicit','/project/ruletaker-cwa-explicit')
  .add_local_file(ROOT/'outputs/autoresearch/ruletaker-cwa-cpu.json','/project/ruletaker-cwa-cpu.json')
  .add_local_file(ROOT/'outputs/autoresearch/ruletaker-cwa-runner-cpu.json','/project/ruletaker-cwa-runner-cpu.json'))
app=modal.App(APP_NAME)
def validate():
 from training_data import digest
 proof=json.loads(Path('/project/ruletaker-cwa-runner-cpu.json').read_text())
 if proof['status']!='passed':raise RuntimeError('Passed CPU runner tests required')
 for name,expected in proof['source_sha256'].items():
  if digest(Path('/project')/name)!=expected:raise ValueError('Tested repair runner changed: '+name)
 dataset=json.loads(Path('/project/ruletaker-cwa-cpu.json').read_text())
 if dataset['status']!='passed' or digest('/project/ruletaker-cwa-explicit/dataset.jsonl')!=dataset['dataset_sha256']:raise ValueError('CPU-tested repair dataset changed')
 return proof
@app.cls(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3600,retries=0,min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=IDLE,volumes={'/data':data,'/models':models.read_only(),'/results':results},secrets=[secret])
class RepairWorker:
 @modal.enter()
 def setup(self):
  import torch
  from resident_backbone import BackbonePool
  from kernel_runtime import configure_optimized_runtime
  from training_data import digest
  validate();configure_optimized_runtime('both');torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
  self.pool=BackbonePool();self.failed=False;self.identity={'worker_id':uuid.uuid4().hex,'worker_source_sha256':digest(__file__),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()}
 @modal.method()
 def ping(self):
  if time.time()+IDLE+60>=DEADLINE:raise TimeoutError('Reservation expiry')
  return dict(self.identity,failed=self.failed,model_loads=self.pool.loads,model_hits=self.pool.hits)
 @modal.method()
 def run(self,run_id,commit):
  from training_data import validate_run_id
  from run_ruletaker_cwa import run
  validate_run_id(run_id)
  if self.failed:raise RuntimeError('Prior failure requires diagnosis')
  data.reload();results.reload()
  try:
   with self.pool.activate():value=run(Path('/results')/run_id,commit,self.identity,results.commit,data.commit,DEADLINE)
   return dict(value,model_loads=self.pool.loads,model_hits=self.pool.hits)
  except Exception:self.failed=True;raise
@app.function(image=image,cpu=2,memory=4096,timeout=120,retries=0,scaledown_window=2)
def check_imports():
 from run_ruletaker_cwa import run
 from training_data import digest
 proof=validate()
 return {'status':'passed','source_sha256':proof['source_sha256'],'worker_source_sha256':digest(__file__),'GPU_allocated':False,'entrypoint':'RepairWorker.run','launch_state':'not deployed; original source grid and single-GPU handoff must finish first'}
@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/ruletaker-cwa-worker-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior import receipt')
 value=check_imports.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
