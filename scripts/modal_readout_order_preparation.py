"""CPU-only protocol tests and actual-tokenizer preflight for reverse pooling."""
import json,subprocess,hashlib
from pathlib import Path
import modal,sys
if Path("/project/scripts").is_dir():sys.path.insert(0,"/project/scripts")
from modal_pretrained_readout import ROOT,ENV,image,models,results
app=modal.App('ac-worktrial-readout-order-preparation')
@app.function(image=image,cpu=4,memory=16384,timeout=900,retries=0,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results.read_only()})
def validate():
 import sys
 sys.path.insert(0,'/project/scripts')
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_readout_order_pool','test_readout_order_encoding','test_pretrained_readout','test_pretrained_readout_pipeline','test_pretrained_readout_io','test_head_probe_io','test_checkpoint_selection'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-14000:])
 from transformers import AutoProcessor
 from training_data import load_splits,digest
 from run_baselines import load_panel
 from evaluate_pretrained_readout import prepare_decisions
 cfg=json.loads(Path('/project/configs/e4-qwen38-27b-v3.json').read_text());tokenizer=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 splits,manifest=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json')
 reference=json.loads(Path('/project/readout-preflight.json').read_text());sets={}
 for role in ['calibration','dev','test']:
  rows=load_panel() if role=='test' else splits[role];sets[role]={}
  for order in ['native','reverse']:
   prepared=prepare_decisions(tokenizer,rows,training_sources=role!='test',option_order=order);fingerprint=hashlib.sha256()
   for item in prepared:fingerprint.update(json.dumps({'id':item['id'],**item['encoding']},sort_keys=True,separators=(',',':')).encode())
   entry={'records':len(prepared),'encodings_sha256':fingerprint.hexdigest(),'max_tokens':max(len(x['encoding']['input_ids']) for x in prepared),'total_tokens':sum(len(x['encoding']['input_ids']) for x in prepared)}
   if order=='native' and (entry['encodings_sha256']!=reference['sets'][role]['encodings_sha256'] or len(prepared)!=reference['sets'][role]['expected']):raise ValueError('Native baseline input changed: '+role)
   sets[role][order]=entry
 names=['scripts/pretrained_readout.py','scripts/evaluate_pretrained_readout.py','scripts/readout_order_pool.py','scripts/modal_readout_order_preparation.py','tests/test_readout_order_pool.py','tests/test_readout_order_encoding.py']
 return {'status':'passed','tests':p.stderr,'backbone_id':cfg['backbone_id'],'revision':cfg['backbone_revision'],'sets':sets,'manifest_sha256':digest('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json'),'source_sha256':{name:digest(Path('/project')/name) for name in names},'scope':'Only CPU tokenization and contract fixtures. Native encodings match original locked fingerprints exactly. Reverse inputs have no gold/audit channel or truncation; no GPU inference, fitting or test scoring.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/readout-order-preparation.json'
 if out.exists():raise FileExistsError('Preserve prior receipt')
 value=validate.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
