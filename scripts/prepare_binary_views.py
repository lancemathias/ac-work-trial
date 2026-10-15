"""Matched Choice replay / Noul entailment views of audited ANLI train data."""
import argparse,collections,copy,hashlib,json,shutil
from pathlib import Path
INSTRUCTION='Does the premise entail the hypothesis? Answer true if the hypothesis follows from the premise, and false if the premise contradicts it.'
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).open()]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def candidates(splits,seed,cap):
 train=splits['train'];held=[r for role in ['dev','calibration'] for r in splits[role]]
 held_groups={r['group'] for r in held};held_states={canonical(r['state']) for r in held}
 selected={'entailment':[],'contradiction':[]}
 for row in train:
  if row['group'] in held_groups or canonical(row['state']) in held_states:raise ValueError('Parent train/held-out leakage')
  if row['suite']!='ANLI':continue
  if row['question']['type']!='choice':raise ValueError('Expected native ANLI Choice')
  meaning=row['question']['criteria'][row['label']].strip().lower()
  if meaning not in {'entailment','contradiction','neutral'}:raise ValueError('Unknown native ANLI label')
  if meaning in selected:selected[meaning].append(row)
 count=min(cap,*(len(v) for v in selected.values()))
 if not count:raise ValueError('No balanced ANLI views available')
 rows=[]
 for meaning,items in selected.items():
  items.sort(key=lambda r:hashlib.sha256(canonical([seed,r['id']]).encode()).hexdigest())
  for row in items[:count]:rows.append((row,meaning))
 return sorted(rows,key=lambda pair:pair[0]['id'])
def views(selected,kind):
 result=[]
 for parent,meaning in selected:
  row=copy.deepcopy(parent);row['id']='ANLI:binary-view:'+parent['id']
  row['source']['view_parent_id']=parent['id'];row['source']['view_rule']='matched_choice_replay' if kind=='choice_replay' else 'human_entailment_true_contradiction_false; neutral excluded'
  if kind=='noul':row['question']={'type':'noul','instructions':INSTRUCTION};row['label']=meaning=='entailment'
  elif kind!='choice_replay':raise ValueError('Unknown view kind')
  result.append(row)
 return result
def prepare(parent,audit,destination,expected_manifest_sha256,seed=20261014,cap=256):
 parent=Path(parent);destination=Path(destination)
 if destination.exists():raise FileExistsError(destination)
 if sha(parent/'manifest.json')!=expected_manifest_sha256:raise ValueError('Parent split manifest changed')
 audit_file=Path(audit);manifest=json.loads((parent/'manifest.json').read_text());audit=json.loads(audit_file.read_text())
 if not audit['approved'] or audit['dataset_sha256']!=manifest['source_dataset_sha256']:raise ValueError('Unaudited parent dataset')
 splits={}
 for role in ['train','dev','calibration']:
  p=parent/manifest['splits'][role]['points']['path']
  if sha(p)!=manifest['splits'][role]['points']['sha256']:raise ValueError('Parent partition changed')
  splits[role]=read(p)
 selected=candidates(splits,seed,cap);destination.mkdir(parents=True)
 receipts={}
 for kind in ['choice_replay','noul']:
  out=destination/kind;out.mkdir();added=views(selected,kind);allrows=splits['train']+added+splits['dev']+splits['calibration']
  if len({r['id'] for r in allrows})!=len(allrows):raise ValueError('Duplicate IDs')
  dataset=out/'dataset.jsonl';dataset.write_text(''.join(canonical(r)+'\n' for r in allrows))
  (out/'additions.jsonl').write_text(''.join(canonical(r)+'\n' for r in added))
  m=copy.deepcopy(manifest);m['source_dataset_sha256']=sha(dataset);m['split_rule']='Preserve exact E3 dev/calibration bytes; original train plus balanced ANLI training-only views; no re-splitting'
  proof={'approved':True,'dataset_sha256':sha(dataset),'parent_dataset_sha256':manifest['source_dataset_sha256'],'parent_audit_sha256':sha(audit_file),'basis':'All states and native sources inherited from the full-pool audited parent. Added views only reference verified original ANLI training rows; native human entailment/contradiction labels map deterministically; neutral excluded. No benchmark task enters any role.','view_kind':kind,'added_records':len(added),'gold_counts':dict(collections.Counter(str(r['label']) for r in added))}
  write(out/'hygiene-report.json',proof);m['hygiene']={'approved':True,'report':{'path':'hygiene-report.json','sha256':sha(out/'hygiene-report.json')}}
  for role in ['train','dev','calibration']:
   target=out/(role+'.jsonl');original=parent/manifest['splits'][role]['points']['path']
   target.write_bytes(original.read_bytes()+(''.join(canonical(r)+'\n' for r in added).encode() if role=='train' else b''))
   m['splits'][role]['count']=len(splits[role])+(len(added) if role=='train' else 0);m['splits'][role]['points']={'path':target.name,'sha256':sha(target)}
  write(out/'manifest.json',m)
  receipts[kind]={'dataset_sha256':sha(dataset),'additions_sha256':sha(out/'additions.jsonl'),'split_manifest_sha256':sha(out/'manifest.json'),'records':len(allrows),'added_records':len(added),'splits':{role:m['splits'][role]['points']['sha256'] for role in splits}}
 receipt={'state':'prepared_candidate_not_launched','seed':seed,'cap_per_class':cap,'selected_parent_ids':[r['id'] for r,_ in selected],'parent_split_manifest_sha256':expected_manifest_sha256,'conditions':receipts,'comparison':'Same selected parent IDs, number of added decisions and optimizer updates; Choice replay controls for extra exposure versus Noul typed views. E4/E5 results determine whether to launch.'}
 write(destination/'release.json',receipt);return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--audit',required=True);p.add_argument('--destination',required=True);p.add_argument('--expected-manifest-sha256',required=True);a=p.parse_args()
 result=prepare(a.parent,a.audit,a.destination,a.expected_manifest_sha256)
 print(json.dumps({k:v for k,v in result.items() if k!='selected_parent_ids'},indent=2))
