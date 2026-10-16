"""Pinned CPU generation/cache contracts plus production input fingerprints."""
import json,sys,subprocess,hashlib,inspect
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image as base_image,models,results
image=base_image
if modal.is_local():image=image.add_local_file(ROOT/'outputs/autoresearch/thinking-readout-preflight.json','/project/thinking-preflight.json')
app=modal.App('ac-worktrial-bounded-thinking-check')
@app.function(image=image,cpu=4,memory=16384,timeout=900,retries=0,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results.read_only()})
def check():
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_bounded_thinking','test_pretrained_readout','test_pretrained_readout_pipeline','test_readout_order_pool','test_readout_order_encoding'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:return {'status':'failed','phase':'CPU_generation_contracts','tests':p.stderr,'returncode':p.returncode}
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from bounded_thinking_readout import prepare
 from training_data import load_splits,digest
 from run_baselines import load_panel
 settings=json.loads(Path('/project/configs/bounded-thinking128.json').read_text());preflight=json.loads(Path('/project/thinking-preflight.json').read_text())
 if digest('/project/thinking-preflight.json')!=settings['input_preflight_sha256']:raise ValueError('Thinking input preflight changed')
 if 'logits_to_keep' not in inspect.signature(Qwen3_5ForConditionalGeneration.forward).parameters:raise ValueError('Actual GPU wrapper does not support last-token generation logits')
 tok=AutoProcessor.from_pretrained(settings['backbone_id'],revision=settings['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 if tok('</think>',add_special_tokens=False).input_ids!=[preflight['close_think_token_id']]:raise ValueError('Native closing token changed')
 splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json');sets={}
 for role in ['calibration','dev','test']:
  rows=load_panel() if role=='test' else splits[role];items=prepare(tok,rows,settings,training_sources=role!='test');h=hashlib.sha256()
  for item in items:h.update(json.dumps({'id':item['id'],**item['encoding']},sort_keys=True,separators=(',',':')).encode())
  expected=preflight['sets'][role]
  if h.hexdigest()!=expected['thinking_encodings_sha256'] or len(items)!=expected['records']:raise ValueError('Production thinking input differs: '+role)
  sets[role]={'records':len(items),'encodings_sha256':h.hexdigest(),'max_prompt_tokens':max(len(x['encoding']['input_ids']) for x in items),'all_within_amended_cap':all(len(x['encoding']['input_ids'])<=settings['thinking_prompt_cap'] for x in items)}
 names=['scripts/bounded_thinking_readout.py','scripts/modal_bounded_thinking_check.py','tests/test_bounded_thinking.py','configs/bounded-thinking128.json']
 return {'status':'passed','tests':p.stderr,'sets':sets,'source_sha256':{n:digest(Path('/project')/n) for n in names},'actual_conditional_forward_signature':str(inspect.signature(Qwen3_5ForConditionalGeneration.forward)),'scope':'CPU tiny text-Qwen cached generation/full readout and exact production tokenizer inputs; real27B conditional-wrapper GPU smoke and throughput budget still required before a full trial.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/bounded-thinking-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior CPU result')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
 if value['status']!='passed':raise RuntimeError('Bounded-thinking CPU contract failed; GPU launch blocked')
