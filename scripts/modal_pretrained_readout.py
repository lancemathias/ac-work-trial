"""Bounded frozen LM readout, queued after requested data ablations."""
import json,time,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';DEADLINE=1792125869
DEPENDENCY='ab1-fast-accuracy-polarity-seed20261014'
app=modal.App('ac-worktrial-pretrained-readout')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .pip_install_from_requirements(ROOT/'requirements-kernels.txt',extra_options='--no-deps')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs').add_local_dir(ROOT/'tests','/project/tests')
 .add_local_file(ROOT/'outputs/autoresearch/readout-tokenizer-letters-v1.json','/project/readout-preflight.json')
 .add_local_dir(ROOT/'data/manifests','/project/data/manifests').add_local_dir(ROOT/'data/processed','/project/data/processed'))
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
def budget():
 if time.time()+3600>DEADLINE:raise TimeoutError('Insufficient reserved A100 window for bounded readout')
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def cpu_check():
 import sys
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_pretrained_readout','test_pretrained_readout_pipeline','test_checkpoint_selection','test_training','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-12000:])
 return {'status':'passed','tests':p.stderr}
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3600,retries=0,max_containers=1,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results},secrets=[secret])
def evaluate(run_id,commit):
 budget()
 import sys
 sys.path.insert(0,'/project/scripts')
 from evaluate_pretrained_readout import evaluate as run
 try:return run(Path('/results')/run_id,commit,results.commit)
 finally:results.commit()
@app.function(image=image,cpu=.25,memory=1024,timeout=14*3600,retries=0,max_containers=1,volumes={'/results':results})
def pipeline(run_id,commit):
 out=Path('/results')/(run_id+'-pipeline');out.mkdir(exist_ok=False)
 def status(phase,**kw):
  (out/'status.json').write_text(json.dumps({'phase':phase,'run_id':run_id,'commit':commit,**kw},indent=2)+'\n');results.commit()
 try:
  status('cpu_validation');validation=cpu_check.remote();(out/'cpu-validation.json').write_text(json.dumps(validation,indent=2)+'\n');results.commit()
  status('waiting_for_source_grid_and_polarity')
  while True:
   budget();results.reload();state=Path('/results')/(DEPENDENCY+'-pipeline')/'status.json'
   if state.exists():
    phase=json.loads(state.read_text())['phase']
    if phase=='failed':raise RuntimeError('Dependency failed; halt readout queue for diagnosis')
    if phase=='complete':break
   time.sleep(30)
  status('frozen_readout');result=evaluate.remote(run_id,commit);status('complete',result=result);return result
 except Exception as exc:status('failed',error=type(exc).__name__,message=str(exc));raise
@app.local_entrypoint()
def main(run_id:str):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 preflight=json.loads((ROOT/'outputs/autoresearch/readout-tokenizer-letters-v1.json').read_text())
 if preflight['status']!='passed':raise ValueError('Tokenizer feasibility required')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 print(json.dumps(pipeline.remote(run_id,commit),indent=2))
@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=cpu_check.remote();(ROOT/'outputs/autoresearch/pretrained-readout-pipeline-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
