"""CPU reproduction, immutable-field/split/window and tokenizer checks for CWA repair."""
import json,sys,tempfile,hashlib
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,image as base_image,models
image=base_image
if modal.is_local():
 image=(image.add_local_dir(ROOT/'data/training/ruletaker-cwa-explicit','/project/ruletaker-cwa-explicit')
  .add_local_file(ROOT/'data/training/sample-5000.jsonl','/project/data/training/sample-5000.jsonl')
  .add_local_file(ROOT/'outputs/autoresearch/source-addback-release.json','/project/outputs/autoresearch/source-addback-release.json')
  .add_local_file(ROOT/'outputs/model-artifacts/ab1-math-accuracy-seed20261014-ruletaker-training/config.json','/project/original-ruletaker-trained-config.json'))
app=modal.App('ac-worktrial-ruletaker-semantics-check')
@app.function(image=image,cpu=4,memory=16384,timeout=900,retries=0,scaledown_window=2,volumes={'/models':models.read_only()})
def check():
 from prepare_ruletaker_semantics import prepare,sha,NEW
 from training_data import load_splits
 from train_joint import prepare as encode_rows,epoch_windows
 from transformers import AutoProcessor
 from types import SimpleNamespace
 source=Path('/project/source-addbacks/ruletaker');target=Path('/project/ruletaker-cwa-explicit');cfgpath=Path('/project/configs/source-quality/ruletaker-cwa-explicit.json');cfg=json.loads(cfgpath.read_text())
 with tempfile.TemporaryDirectory() as tmp:
  temp=Path(tmp);proof=prepare(source,temp/'data',Path('/project/data/training/sample-5000.jsonl'),temp/'config.json')
  reproduced=[]
  for p in target.iterdir():
   if p.is_file():
    if p.read_bytes()!=(temp/'data'/p.name).read_bytes():raise ValueError('Non-reproducible CWA artifact: '+p.name)
    reproduced.append(p.name)
  if cfgpath.read_bytes()!=(temp/'config.json').read_bytes():raise ValueError('Non-reproducible config')
 original,_=load_splits(source/'manifest.json');splits,manifest=load_splits(target/'manifest.json')
 if [r['id'] for r in original['train']]!=[r['id'] for r in splits['train']]:raise ValueError('Training IDs/order changed')
 for role in ['dev','calibration']:
  if original[role]!=splits[role] or sha(source/(role+'.jsonl'))!=sha(target/(role+'.jsonl')):raise ValueError('Heldout bytes/content changed')
 if any(len(r['targets'])!=1 for values in splits.values() for r in values):raise ValueError('Expected one decision per request')
 def windows(rows):
  dummy=[(row,SimpleNamespace(questions=[None])) for row in rows]
  return [[rows[i]['id'] for i in batch] for epoch in range(5) for batch in epoch_windows(dummy,cfg['seed'],epoch,cfg['effective_batch_decisions'])]
 old_windows,new_windows=windows(original['train']),windows(splits['train'])
 if old_windows!=new_windows or len(new_windows)!=510:raise ValueError('Optimizer windows differ')
 oldcfg=json.loads(Path('/project/original-ruletaker-trained-config.json').read_text())['config'];variable={'dataset','run_name','purpose'}
 if {k:v for k,v in oldcfg.items() if k not in variable}!={k:v for k,v in cfg.items() if k not in variable}:raise ValueError('Training recipe differs')
 changed=[r for r in splits['train'] if r['source']=='RuleTaker']
 if len(changed)!=250 or any(not r['request']['questions']['q']['instructions'].startswith(NEW) for r in changed):raise ValueError('Missing uniform CWA instruction')
 tok=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer;encoded=encode_rows(changed,tok,cfg['max_input_tokens']);fingerprint=hashlib.sha256()
 for row,e in encoded:fingerprint.update(json.dumps({'id':row['id'],'input_ids':e.input_ids},sort_keys=True,separators=(',',':')).encode())
 names=['scripts/prepare_ruletaker_semantics.py','scripts/modal_ruletaker_semantics_check.py','configs/source-quality/ruletaker-cwa-explicit.json']
 return {'status':'passed','reproduced_files':reproduced,'config_byte_identical':True,'loader_records':{k:len(v) for k,v in splits.items()},'all510_optimizer_windows_identical':True,'heldout_bytes_exact':True,'only250_instruction_prefixes_changed':True,'native_labels_preserved':True,'unambiguous_synthetic_semantic_cases':proof['unambiguous_synthetic_positive_Horn_cases'],'changed_requests_encoded':len(encoded),'max_changed_input_tokens':max(len(e.input_ids) for _,e in encoded),'changed_encoding_sha256':fingerprint.hexdigest(),'dataset_sha256':manifest['source_dataset_sha256'],'source_sha256':{n:sha(Path('/project')/n) for n in names},'scope':'CPU data/loader/tokenizer only, no frozen feature extraction or new training. Fresh250CWA feature cache required; reuse original v3base features only, never old ambiguous Rule cache.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/ruletaker-cwa-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior CPU proof')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
