"""CPU-only loader/tokenizer and exact native-label matching gate."""
import json,collections,sys
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_secondary_data_check import ROOT,ENV,image as base_image,models,validate as validate_data
image=base_image
if modal.is_local():image=(image.add_local_dir(ROOT/'data/training/boolean-controls','/project/boolean-controls').add_local_file(ROOT/'outputs/autoresearch/source-addback-release.json','/project/first-source-release.json'))
app=modal.App('ac-worktrial-boolean-control-validation')
@app.function(image=image,cpu=4,memory=16384,timeout=600,retries=0,scaledown_window=2,volumes={'/models':models.read_only()})
def validate():
 from training_data import digest
 root=Path('/project/boolean-controls');release=json.loads((root/'release.json').read_text());first=json.loads(Path('/project/first-source-release.json').read_text())
 if release['first_release_sha256']!=digest('/project/first-source-release.json'):raise ValueError('Different first-round conditions')
 result=validate_data.local(str(root),'/project/configs/boolean-controls');matches={}
 for name,spec in release['conditions'].items():
  rows=[json.loads(l) for l in (root/name/'additions.jsonl').open()];counts=dict(collections.Counter(str(r['label']) for r in rows))
  if counts!=first['conditions'][spec['matched_original_condition']]['added_label_counts'] or any(r['question']['type']!='noul' or r['suite']!='MultiRC' for r in rows):raise ValueError('Type or native class mismatch')
  matches[name]=counts
 result['native_class_matches']=matches;result['scope']='Prepared follow-up data only; actual loader/tokenizer, recipe, held-outs and original training-label counts verified. No GPU allocation.';return result
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=validate.remote();(ROOT/'outputs/autoresearch/boolean-control-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
