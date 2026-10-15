"""Pinned optimized runtime; compare actual frozen Qwen kernels on one A100."""
import json,time,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';DEADLINE=1792125869
app=modal.App('ac-worktrial-optimized-kernel-validation')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .pip_install_from_requirements(ROOT/'requirements-kernels.txt',extra_options='--no-deps')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs')
 .add_local_dir(ROOT/'tests','/project/tests'))
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV);results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def cpu_check():
 import sys,importlib.metadata as m,importlib.util
 sys.path.insert(0,'/project/scripts')
 from kernel_runtime import configure_optimized_runtime
 runtime=configure_optimized_runtime('both')
 versions={n:m.version(n) for n in ['torch','triton','transformers','causal-conv1d','fla-core','flash-linear-attention','einops']}
 if versions['torch']!='2.9.1' or versions['transformers']!='5.17.0' or versions['triton']!='3.5.1':raise ValueError('Pinned runtime changed')
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_cache_composition','test_checkpoint_selection','test_training','test_cached_training','test_training_runtime','test_backbone_attention','test_adamw_groups','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-12000:])
 return {'status':'passed','versions':versions,'configured_future_runtime':runtime,'tests':p.stderr}
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=1800,retries=0,max_containers=1,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results})
def benchmark(run_id,commit,kernel_set):
 if time.time()+1800>DEADLINE:raise TimeoutError('Insufficient A100 reservation')
 import sys
 sys.path.insert(0,'/project/scripts')
 from benchmark_kernels import run
 try:
  if kernel_set=='isolate':
   import gc,torch
   summaries={}
   for kind in ['conv_only','linear_only']:
    summaries[kind]=run(run_id+'-'+kind,commit,results.commit,kind)
    gc.collect();torch.cuda.empty_cache()
   return summaries
  return run(run_id,commit,results.commit,kernel_set)
 finally:results.commit()
@app.local_entrypoint()
def main(run_id:str='optimized-kernels-v1',cpu_only:bool=False,kernel_set:str='both'):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 cpu=cpu_check.remote();(ROOT/'outputs/autoresearch/optimized-kernels-cpu.json').write_text(json.dumps(cpu,indent=2)+'\n');print(json.dumps(cpu,indent=2))
 if not cpu_only:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();print(json.dumps(benchmark.remote(run_id,commit,kernel_set),indent=2))
