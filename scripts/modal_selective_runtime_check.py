"""CPU-only synthetic scoring integration; no result volume or benchmark data mount."""
import json,sys,subprocess
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image
app=modal.App('ac-worktrial-selective-runtime-cpu')
@app.function(image=image,cpu=2,memory=8192,timeout=600,retries=0,scaledown_window=2)
def check():
 from training_data import digest
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_selective_thinking','test_selective_development','test_selective_test_gate','test_selective_cached_runtime','test_fewshot_capacity'],cwd='/project/tests',capture_output=True,text=True)
 return {'status':'passed' if p.returncode==0 else 'failed','tests':p.stderr,'source_sha256':{n:digest(Path('/project')/n) for n in ['scripts/selective_thinking.py','scripts/assess_selective_thinking.py','scripts/apply_selective_cached_test.py','scripts/fewshot_capacity.py','scripts/fewshot_smoke.py','tests/test_selective_cached_runtime.py']},'scope':'Synthetic2111-row cached scoring integration and refusal paths,9policy/dev/access gates,2capacity selection checks. No real benchmark access, result-volume mount or GPU.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/selective-cached-runtime-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior validation')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
 if value['status']!='passed':raise RuntimeError('Synthetic CPU runtime validation failed')
