"""CPU-only tests for prospective same-record repair orchestration."""
import json,sys,subprocess
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image
app=modal.App('ac-worktrial-ruletaker-runner-cpu')
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def check():
 from training_data import digest
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_ruletaker_cwa_runner','test_cache_composition','test_head_probe_io'],cwd='/project/tests',capture_output=True,text=True)
 names=['scripts/run_ruletaker_cwa.py','tests/test_ruletaker_cwa_runner.py']
 return {'status':'passed' if p.returncode==0 else 'failed','tests':p.stderr,'source_sha256':{n:digest(Path('/project')/n) for n in names},'scope':'CPU mocked orchestration, actual cache composition and immutable IO contracts. No real feature extraction, GPU training or benchmark performance.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/ruletaker-cwa-runner-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior result')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
 if value['status']!='passed':raise RuntimeError('Repair runner checks failed')
