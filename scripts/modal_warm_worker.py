"""One shared A100 worker, retaining model weights and Triton compilation."""
import json,time,uuid
from pathlib import Path
import modal
import sys
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,DEADLINE,image as source_image,data,models,results,secret
APP_NAME='ac-worktrial-warm-research'
IDLE_SECONDS=1200
image=source_image
if modal.is_local():
 image=(source_image
  .add_local_file(ROOT/'data/training/polarity-dev-probe.jsonl','/project/polarity-probe.jsonl')
  .add_local_file(ROOT/'outputs/model-artifacts/e3-medium-v3-seed20261014-e7dd4bb-splits/dev.jsonl','/project/source-dev.jsonl')
  .add_local_file(ROOT/'outputs/training/e3-medium-v3-seed20261014-e7dd4bb/config.json','/project/e3-provenance.json')
  .add_local_file(ROOT/'outputs/autoresearch/readout-tokenizer-letters-v1.json','/project/readout-preflight.json'))
app=modal.App(APP_NAME)
def require_window(seconds):
 if time.time()+seconds+IDLE_SECONDS+60>DEADLINE:raise TimeoutError('Insufficient reservation including warm-container expiry')
@app.cls(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3600,retries=0,
 min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=IDLE_SECONDS,
 volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
class GPUWorker:
 @modal.enter()
 def setup(self):
  import sys,torch,subprocess
  sys.path.insert(0,'/project/scripts')
  from kernel_runtime import configure_optimized_runtime
  from resident_backbone import BackbonePool
  torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
  self.kernels=configure_optimized_runtime('both');self.pool=BackbonePool();self.worker_id=uuid.uuid4().hex;self.failed=False
  self.gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()
 @modal.method()
 def ping(self):
  require_window(0)
  return {'worker_id':self.worker_id,'gpu':self.gpu,'loads':self.pool.loads,'hits':self.pool.hits,'backbone':self.pool.key,'failed':self.failed}
 @modal.method()
 def run_job(self,kind,job_id,arguments):
  require_window(3600)
  if self.failed:raise RuntimeError('Prior job failed; diagnose before replacing this worker')
  from training_data import validate_run_id
  validate_run_id(job_id)
  root=Path('/results/warm-worker-jobs');root.mkdir(exist_ok=True);receipt=root/(job_id+'.json')
  if receipt.exists():raise FileExistsError('Preserve existing worker receipt: '+job_id)
  began=time.monotonic();before=self.pool.loads
  try:
   with self.pool.activate():
    if kind=='smoke':
     import torch,hashlib
     from resident_backbone import get_model
     cfg=json.loads(Path('/project/configs/e3-medium-v3.json').read_text());model=get_model(cfg)
     ids=torch.tensor([[1,2,3,4]],device='cuda');text=model.model
     if hasattr(text,'language_model'):text=text.language_model
     with torch.no_grad():hidden=text(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
     value={'hidden_sha256':hashlib.sha256(hidden.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
    elif kind=='extract_sources':
     from modal_source_addbacks import extract_union
     value=extract_union.local()
    elif kind=='source_condition':
     from modal_source_addbacks import train_and_benchmark
     value=train_and_benchmark.local(**arguments)
    elif kind=='polarity':
     from modal_polarity_diagnostic import evaluate_heads
     value=evaluate_heads.local(**arguments)
    elif kind=='readout':
     from evaluate_pretrained_readout import evaluate
     value=evaluate(Path('/results')/arguments['run_id'],arguments['commit'],results.commit)
    else:raise ValueError('Unsupported warm-worker job: '+kind)
   result={'status':'complete','worker_id':self.worker_id,'gpu':self.gpu,'job_id':job_id,'kind':kind,'elapsed_seconds':time.monotonic()-began,'model_loads_before':before,'model_loads_after':self.pool.loads,'model_reuse_hits':self.pool.hits,'result':value}
   receipt.write_text(json.dumps(result,indent=2)+'\n');results.commit();return result
  except Exception as exc:
   self.failed=True;receipt.write_text(json.dumps({'status':'failed','worker_id':self.worker_id,'error':type(exc).__name__,'message':str(exc)},indent=2)+'\n');results.commit();raise
