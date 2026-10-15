"""CPU-only checks for prospective thinking smoke; never allocates a GPU."""
import json,sys,subprocess
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image as base_image
app=modal.App('ac-worktrial-thinking-smoke-cpu')
image=base_image
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def check():
 from bounded_thinking_smoke import smoke
 from evaluate_bounded_thinking import evaluate
 from training_data import digest
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_thinking_smoke_plan','test_bounded_thinking','test_bounded_thinking_runtime'],cwd='/project/tests',capture_output=True,text=True)
 names=['scripts/bounded_thinking_smoke.py','scripts/thinking_smoke_plan.py','tests/test_thinking_smoke_plan.py','scripts/evaluate_bounded_thinking.py','tests/test_bounded_thinking_runtime.py']
 return {'status':'passed' if p.returncode==0 else 'failed','tests':p.stderr,'source_sha256':{n:digest(Path('/project')/n) for n in names},'scope':'CPU generation contracts, smoke-selection and reservation arithmetic, import check only. Includes evaluator dev reject/pass branch tests; does not execute real27B GPU smoke or validate full smoke runner branches.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/thinking-smoke-capacity-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior CPU receipt')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
 if value['status']!='passed':raise RuntimeError('CPU smoke preparation failed')
