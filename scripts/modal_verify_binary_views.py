"""Validate both candidate datasets through the actual training input loader."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
app=modal.App('ac-worktrial-binary-view-data-validation')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('jsonschema==4.25.1')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs')
 .add_local_dir(ROOT/'data/training/e6-candidate','/candidate'))
@app.function(image=image,cpu=1,memory=1024,timeout=300,retries=0,scaledown_window=2)
def verify():
 import sys,collections
 sys.path.insert(0,'/project/scripts')
 from training_data import load_splits,digest
 root=Path('/candidate');data={};result={}
 for name in ['choice_replay','noul']:
  splits,manifest=load_splits(root/name/'manifest.json');data[name]=splits
  result[name]={'counts':{k:len(v) for k,v in splits.items()},'train_type_counts':dict(collections.Counter(r['request']['questions']['q']['type'] for r in splits['train'])),'dev_sha256':digest(root/name/'dev.jsonl'),'calibration_sha256':digest(root/name/'calibration.jsonl')}
 for role in ['dev','calibration']:
  if data['choice_replay'][role]!=data['noul'][role]:raise ValueError('Held-out partitions changed')
 for role in ['train','dev','calibration']:
  a,b=data['choice_replay'][role],data['noul'][role]
  if [r['id'] for r in a]!=[r['id'] for r in b]:raise ValueError('Unmatched record order')
  if [r['request']['state'] for r in a]!=[r['request']['state'] for r in b]:raise ValueError('Unmatched source states')
 return {'status':'passed','conditions':result,'held_out_partitions_identical':True,'same_native_ids_and_order':True,'source_states_identical':True,'validation':'Actual load_splits validates JSON schema, allowed sources, typed targets, hashes and train/dev/calibration disjointness.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=verify.remote();(ROOT/'outputs/autoresearch/e6-data-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
