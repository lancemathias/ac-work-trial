"""CPU integration tests for conditional benchmark access in order pooling."""
import json,subprocess,sys,hashlib
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image
app=modal.App('ac-worktrial-readout-order-runtime-check')
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def validate():
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_readout_order_runtime','test_readout_order_pool','test_readout_order_encoding','test_pretrained_readout','test_pretrained_readout_pipeline','test_pretrained_readout_io','test_head_probe_io','test_checkpoint_selection'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-16000:])
 names=['scripts/evaluate_readout_order.py','scripts/readout_order_pool.py','scripts/pretrained_readout.py','scripts/evaluate_pretrained_readout.py','scripts/head_probe_io.py','scripts/modal_readout_order_runtime_check.py','tests/test_readout_order_runtime.py']
 return {'status':'passed','tests':p.stderr,'source_sha256':{n:hashlib.sha256((Path('/project')/n).read_bytes()).hexdigest() for n in names}}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/readout-order-runtime-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior CPU receipt')
 value=validate.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
