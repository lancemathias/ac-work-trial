"""Run actual training-loader and matched-budget gates before any GPU launch."""
import argparse,collections,json,math
from pathlib import Path
from training_data import load_splits,digest
ROOT=Path(__file__).resolve().parents[1]
def verify(root):
 root=Path(root);release=json.loads((root/'release.json').read_text());baseline=json.loads((ROOT/'outputs/training/e3-medium-v3-seed20261014-e7dd4bb/config.json').read_text())
 if release['parent_manifest_sha256']!=baseline['manifest_sha256']:raise ValueError('Wrong E3 parent manifest')
 result={};held={};addedids=set()
 for name,receipt in release['conditions'].items():
  path=root/name/'manifest.json'
  if digest(path)!=receipt['manifest_sha256']:raise ValueError('Condition manifest changed')
  splits,manifest=load_splits(path)
  counts={role:sum(len(r['targets']) for r in rows) for role,rows in splits.items()}
  if counts!={'train':6512,'dev':809,'calibration':857}:raise ValueError('Unmatched decision counts')
  updates=5*math.ceil(counts['train']/64)
  for role in ['dev','calibration']:
   value=digest(root/name/(role+'.jsonl'));expected=baseline['manifest']['splits'][role]['points']['sha256']
   if value!=expected:raise ValueError('E3 held-out partition changed')
   held[role]=value
  additions=[json.loads(x) for x in (root/name/'additions.jsonl').open()]
  if len(additions)!=250 or len({r['id'] for r in additions})!=250:raise ValueError('Unmatched/duplicate additions')
  if any(r['evaluation_only'] or r['suite'] not in receipt['source_counts'] for r in additions):raise ValueError('Invalid added source')
  result[name]={'status':'passed','decisions':counts,'optimizer_updates':updates,'added_source_counts':dict(collections.Counter(r['suite'] for r in additions)),'added_type_counts':dict(collections.Counter(r['question']['type'] for r in additions)),'all_added_ids_unique':True}
 return {'status':'passed','conditions':result,'held_out_sha256':held,'actual_loader':'Validated schema, hashes, allowed sources, targets and cross-role ID/group/state/request isolation.','scope':'Data preparation and recipe equality only. No model-quality result yet.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');a=p.parse_args();result=verify(a.root);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
