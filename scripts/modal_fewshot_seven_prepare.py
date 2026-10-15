"""Select locked training-only demonstrations and fingerprint all candidate inputs."""
import json,sys,hashlib
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,image as base_image,models
app=modal.App('ac-worktrial-fewshot-seven-cpu')
image=base_image.add_local_file(ROOT/'outputs/autoresearch/experiment-cards.json','/project/experiment-cards.json') if modal.is_local() else base_image
@app.function(image=image,cpu=8,memory=16384,timeout=1800,retries=0,scaledown_window=2,volumes={'/models':models.read_only()})
def prepare():
 from transformers import AutoProcessor
 from training_data import load_splits,digest
 from run_baselines import load_panel
 from pretrained_readout import encode as native_encode,messages,CODES
 from fewshot_readout import encode,gold_option
 manifest=Path('/project/source-addbacks/aegis2/manifest.json');splits,m=load_splits(manifest)
 raw={role:[json.loads(x) for x in (manifest.parent/(role+'.jsonl')).read_text().splitlines()] for role in ['train','dev','calibration']}
 cfg=json.loads(Path('/project/configs/e4-qwen38-27b-v3.json').read_text());tok=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 cards=json.loads(Path('/project/experiment-cards.json').read_text())['cards'];card=next(c for c in cards if c['id']=='pretrained-native-seven-fewshots')
 cells=[('MultiRC','true'),('MultiRC','false'),('Aegis2','true'),('Aegis2','false'),('ANLI','entailment'),('ANLI','contradiction'),('ANLI','neutral')];eligible={cell:[] for cell in cells}
 for row in raw['train']:
  source=row['suite']
  if source not in ['MultiRC','Aegis2','ANLI']:continue
  if row['evaluation_only'] is not False:raise ValueError('Evaluation-only row in training candidate pool')
  label=str(row['label']).lower() if source!='ANLI' else row['question']['criteria'][row['label']].lower()
  cell=(source,label)
  if cell not in eligible:raise ValueError('Unknown source/label cell')
  e=native_encode(tok,row['state'],row['question'],maximum=8192)
  if len(e['input_ids'])<=384:eligible[cell].append((len(e['input_ids']),row['id'],row))
 missing=[str(cell) for cell,values in eligible.items() if not values]
 if missing:return {'status':'infeasible_under_preregistered384_token_filter','missing_cells':missing,'eligible_counts':{s+':'+l:len(v) for (s,l),v in eligible.items()},'GPU_allocated':False}
 chosen=[min(eligible[cell],key=lambda x:(x[0],x[1])) for cell in cells];demos=[x[2] for x in chosen]
 train_by_id={r['id']:r for r in raw['train']};held_ids={r['id'] for role in ['dev','calibration'] for r in raw[role]};held_groups={r['group'] for role in ['dev','calibration'] for r in raw[role]}
 if any(train_by_id[r['id']]!=r or r['id'] in held_ids or r['group'] in held_groups for r in demos):raise ValueError('Demo membership or heldout group violation')
 if len({r['id'] for r in demos})!=7:raise ValueError('Repeated demo ID')
 selected=[]
 for (source,label),(tokens,rid,row) in zip(cells,chosen):
  _,options=messages(row['state'],row['question']);selected.append({'id':rid,'source':source,'label_cell':label,'native_tokens':tokens,'answer_code':CODES[options.index(gold_option(row))],'group':row['group'],'point_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()})
 sets={}
 for role in ['calibration','dev','test']:
  rows=load_panel() if role=='test' else raw[role];h=hashlib.sha256();lengths=[];overheads=[]
  for row in rows:
   e=encode(tok,row['state'],row['question'],demos);native=native_encode(tok,row['state'],row['question']);h.update(json.dumps({'id':row['id'],**e},sort_keys=True,separators=(',',':')).encode());lengths.append(len(e['input_ids']));overheads.append(len(e['input_ids'])-len(native['input_ids']))
  sets[role]={'records':len(rows),'encodings_sha256':h.hexdigest(),'maximum_tokens':max(lengths),'total_tokens':sum(lengths),'added_token_range':[min(overheads),max(overheads)]}
 raw_demo=json.dumps(demos,indent=2,ensure_ascii=False)+'\n'
 return {'status':'passed','demonstrations':demos,'selected':selected,'demonstrations_sha256':hashlib.sha256(raw_demo.encode()).hexdigest(),'sets':sets,'eligible_counts':{s+':'+l:len(v) for (s,l),v in eligible.items()},'train_manifest_sha256':digest(manifest),'parent_hygiene_report_sha256':digest(manifest.parent/'hygiene-report.json'),'preregistered_card_sha256':hashlib.sha256(json.dumps(card,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'source_sha256':{n:digest(Path('/project')/n) for n in ['scripts/fewshot_readout.py','scripts/modal_fewshot_seven_prepare.py']},'GPU_allocated':False,'scope':'Exact7 training memberships and heldout group exclusion, inherited full-corpus hygiene, actual pinned tokenizer and all3777 fingerprints. No model inference, devaccuracy or test scoring; helper contract tests and GPU capacity smoke still required.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/fewshot-seven-preparation.json'
 if out.exists():raise FileExistsError('Preserve prior preparation')
 value=prepare.remote()
 if value['status']=='passed':
  demos=value.pop('demonstrations');folder=ROOT/'data/training/fewshot-seven';folder.mkdir(exist_ok=False);(folder/'demonstrations.json').write_text(json.dumps(demos,indent=2,ensure_ascii=False)+'\n')
 out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
