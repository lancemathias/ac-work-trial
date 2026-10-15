"""CPU-only fixed-demonstration message and capacity contracts."""
import json,subprocess,sys
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image as base_image
image=base_image.add_local_file(ROOT/'data/training/fewshot-seven-quality-v2/demonstrations.json','/project/fewshot-quality-v2/demonstrations.json') if modal.is_local() else base_image
app=modal.App('ac-worktrial-fewshot-runtime-cpu')
@app.function(image=image,cpu=2,memory=8192,timeout=600,retries=0,scaledown_window=2)
def check():
 from training_data import digest
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_fewshot_readout','test_pretrained_readout','test_fewshot_runtime'],cwd='/project/tests',capture_output=True,text=True)
 return {'status':'passed' if p.returncode==0 else 'failed','tests':p.stderr,'source_sha256':{n:digest(Path('/project')/n) for n in ['scripts/fewshot_readout.py','scripts/fewshot_pipeline.py','scripts/evaluate_fewshot.py','tests/test_fewshot_readout.py','tests/test_fewshot_runtime.py']},'scope':'CPU helper contracts plus actual evaluator dev rejection, pass and insufficient test-budget branches with mock model inference. No real GPU capacity validation.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/fewshot-quality-v2-runtime-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior receipt')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
 if value['status']!='passed':raise RuntimeError('Fewshot contract checks failed')
