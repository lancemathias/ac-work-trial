"""CPU-only real-tokenizer feasibility check; no model or benchmark fitting."""
import json,hashlib
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
app=modal.App('ac-worktrial-readout-tokenizer-preflight')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs')
 .add_local_dir(ROOT/'data/manifests','/project/data/manifests').add_local_dir(ROOT/'data/processed','/project/data/processed'))
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2,volumes={'/models':models,'/results':results})
def check():
 import sys,collections
 sys.path.insert(0,'/project/scripts')
 from transformers import AutoProcessor
 from pretrained_readout import encode
 from run_baselines import load_panel,sha
 cfg=json.loads(Path('/project/configs/e4-qwen38-27b-v3.json').read_text())
 tokenizer=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 splitroot=Path('/results/e3-medium-v3-seed20261014-e7dd4bb-splits');manifest=json.loads((splitroot/'manifest.json').read_text());answer={}
 for name in ['dev','calibration','test']:
  if name=='test':rows=load_panel();sourcehash=sha('/project/data/manifests/baseline-panel.json')
  else:
   spec=manifest['splits'][name]['points'];p=splitroot/spec['path'];sourcehash=sha(p)
   if sourcehash!=spec['sha256']:raise ValueError('Split bytes changed')
   rows=[json.loads(s) for s in p.read_text().splitlines()]
  lengths=[];failures=[];counts=collections.Counter();fingerprint=hashlib.sha256()
  for r in rows:
   try:e=encode(tokenizer,r['state'],r['question'],maximum=10**9)
   except ValueError as exc:failures.append({'id':r['id'],'error':str(exc)});continue
   n=len(e['input_ids']);lengths.append(n);counts[len(e['option_ids'])]+=1
   if n>8192:failures.append({'id':r['id'],'error':'overlength; no truncation','tokens':n})
   fingerprint.update(json.dumps({'id':r['id'],**e},sort_keys=True,separators=(',',':')).encode())
  answer[name]={'expected':len(rows),'encoded':len(lengths),'maximum_tokens':max(lengths,default=0),'total_tokens':sum(lengths),'option_count_distribution':dict(counts),'failures':failures,'source_sha256':sourcehash,'encodings_sha256':fingerprint.hexdigest()}
 return {'status':'passed' if not any(v['failures'] for v in answer.values()) else 'infeasible_as_written','backbone_id':cfg['backbone_id'],'revision':cfg['backbone_revision'],'sets':answer,'protocol':'CPU tokenization only, no labels used for prompt choice or score fitting; fixed prompt unchanged. No GPU trial launched.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=check.remote();p=ROOT/'outputs/autoresearch/readout-tokenizer-letters-v1.json'
 if p.exists():raise FileExistsError('Preserve prior receipt')
 p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
