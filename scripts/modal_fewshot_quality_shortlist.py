"""Training-only quality-review shortlist; no model scoring or label changes."""
import json,sys
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,image,models
app=modal.App('ac-worktrial-fewshot-quality-cpu')
@app.function(image=image,cpu=4,memory=8192,timeout=900,retries=0,scaledown_window=2,volumes={'/models':models.read_only()})
def shortlist():
 from transformers import AutoProcessor
 from training_data import load_splits,digest
 from pretrained_readout import encode
 manifest=Path('/project/source-addbacks/aegis2/manifest.json');load_splits(manifest)
 rows=[json.loads(x) for x in (manifest.parent/'train.jsonl').read_text().splitlines()];cfg=json.loads(Path('/project/configs/e4-qwen38-27b-v3.json').read_text());tok=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 cells={key:[] for key in ['Aegis2:true','Aegis2:false','ANLI:entailment','ANLI:contradiction']}
 for r in rows:
  source=r['suite']
  if source not in ['Aegis2','ANLI']:continue
  label=str(r['label']).lower() if source=='Aegis2' else r['question']['criteria'][r['label']].lower();key=source+':'+label
  if key not in cells:continue
  if source=='Aegis2' and (not isinstance(r['state'],str) or len(r['state'].split())<12):continue
  if r['evaluation_only'] is not False:raise ValueError('Nontraining source row')
  n=len(encode(tok,r['state'],r['question'])['input_ids'])
  if n<=384:cells[key].append({'native_tokens':n,'id':r['id'],'row':r})
 return {'status':'shortlisted_for_manual_semantic_review','train_manifest_sha256':digest(manifest),'eligible_counts':{k:len(v) for k,v in cells.items()},'shortlists':{k:sorted(v,key=lambda x:(x['native_tokens'],x['id']))[:20] for k,v in cells.items()},'scope':'Native training labels unchanged; complete-request word filter forAegis and actual token counts. No final selection, no model inference, no performance metric.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 folder=ROOT/'data/training/fewshot-seven-quality-v2';folder.mkdir(exist_ok=False)
 value=shortlist.remote();(folder/'shortlist.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='shortlists'},indent=2))
