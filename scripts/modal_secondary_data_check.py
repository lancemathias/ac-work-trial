"""CPU-only real loader/tokenizer gate for prepared, unqueued data candidates."""
import json
from pathlib import Path
import modal,sys
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,image as base_image,models
image=base_image
if modal.is_local():image=image.add_local_dir(ROOT/'data/training/source-addbacks-phase2','/project/secondary-addbacks')
app=modal.App('ac-worktrial-secondary-data-validation')
@app.function(image=image,cpu=4,memory=16384,timeout=600,retries=0,scaledown_window=2,volumes={'/models':models.read_only()})
def validate(data_root='/project/secondary-addbacks',config_root='/project/configs/source-addbacks-phase2'):
 import sys,collections
 sys.path.insert(0,'/project/scripts')
 from verify_source_addbacks import verify
 from transformers import AutoProcessor
 from jsonschema import Draft202012Validator
 from benchmark_training_data import adapt_training_point
 from vendor.clef_joint_schema import encode_record
 from training_data import digest
 root=Path(data_root);release=json.loads((root/'release.json').read_text());checked=verify(root)
 cfg=json.loads(Path('/project/configs/e3-medium-v3.json').read_text());tokenizer=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 validator=Draft202012Validator(json.loads(Path('/project/configs/normalized-training.schema.json').read_text()));union=root/'union-additions.jsonl'
 if digest(union)!=release['union_sha256']:raise ValueError('Union changed')
 lengths=[];types=collections.Counter()
 for line in union.open():
  point=json.loads(line);row=adapt_training_point(point,validator)
  e=encode_record(tokenizer,dict(row['request'],id=row['id']),max_length=10**9)
  if len(e.questions)!=1 or len(e.input_ids)>8192:raise ValueError('Require one in-range decision; no truncation')
  lengths.append(len(e.input_ids));types[point['question']['type']]+=1
 for name,receipt in release['conditions'].items():
  candidate=json.loads(Path(config_root,name+'.json').read_text())
  if candidate['dataset']['sha256']!=receipt['dataset_sha256'] or candidate['selection_metric']!='accuracy':raise ValueError('Candidate configuration differs')
  for key in ['backbone_id','backbone_revision','head','seed','epochs','learning_rate','weight_decay','effective_batch_decisions','microbatch_records','adamw_betas','adamw_eps','adamw_decay_policy']:
   if candidate[key]!=cfg[key]:raise ValueError('Recipe mismatch: '+key)
 return {'status':'passed','data':checked,'union_records':len(lengths),'union_types':dict(types),'maximum_input_tokens':max(lengths),'total_input_tokens':sum(lengths),'backbone_id':cfg['backbone_id'],'revision':cfg['backbone_revision'],'scope':'CPU schema, isolation, counts, recipe and actual tokenizer only. No GPU model loaded; all candidates remain unqueued.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=validate.remote();(ROOT/'outputs/autoresearch/secondary-source-data-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
