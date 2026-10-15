"""Read-only feasibility audit for later requested source add-backs."""
import argparse,collections,json
from pathlib import Path
from prepare_source_addbacks import read,sha,normalize,context_keys,benchmark_index
SOURCES=['SST2','PubMedQA','MCScript2','MCTest','WIQA','ARC','SciQ','GSM8K','HelpSteer2']
def audit(root,parent):
 root=Path(root);parent=Path(parent);manifest=json.loads((root/'data/training/manifests/typed-5000.json').read_text())
 native_path=root/'data/training/sample-5000.jsonl';typed_path=root/'data/training/typed-5000.jsonl'
 if sha(native_path)!=manifest['input_sha256'] or sha(typed_path)!=manifest['output_sha256']:raise ValueError('Pinned v1 bytes changed')
 native={r['id']:r for r in read(native_path)};typed=read(typed_path);pm=json.loads((parent/'manifest.json').read_text());parentrows=[]
 for role in ['train','dev','calibration']:
  path=parent/pm['splits'][role]['points']['path']
  if sha(path)!=pm['splits'][role]['points']['sha256']:raise ValueError('Parent split changed')
  parentrows.extend(read(path))
 ids={r['id'] for r in parentrows};groups={r['group'] for r in parentrows};contexts={k for r in parentrows for k in context_keys(r['state'])}
 hit,pools=benchmark_index();benchmark_names={p['suite'] for p in pools};answer={}
 for source in SOURCES:
  if source in benchmark_names:raise ValueError('Whole benchmark task excluded')
  counts=collections.Counter();labels=collections.Counter();ordinal=collections.defaultdict(collections.Counter);errors=collections.Counter();eligible=[]
  for t in [t for t in typed if t['source']==source]:
   n=native[t['id']];counts['available_v1_records']+=1
   if source=='HelpSteer2':
    if n.get('source_config')!='ratings':
     counts['existing_preference_records_not_ordinal']+=1;continue
    if set(t['field_types'].values())!={'score'} or len(t['gold'])!=5:raise ValueError('Unexpected ordinal fields')
    content=json.loads(t['model_payload']['input'])
    row={'id':t['id'],'group':t['family_id'],'state':content['state'],'question':{'type':'score','instructions':content['question'],'criteria':['0','1','2','3','4']}}
   else:
    try:row=normalize(n,t)
    except (TypeError,ValueError,KeyError) as exc:
     counts['requires_explicit_adapter_review']+=1;errors[type(exc).__name__+': '+str(exc)]+=1;continue
   if row['id'] in ids or row['group'] in groups or context_keys(row['state'])&contexts:counts['parent_overlap_rejected']+=1;continue
   match=hit(row)
   if match:counts['benchmark_text_overlap_rejected']+=1;continue
   eligible.append(row['id']);counts['eligible_records']+=1
   if source=='HelpSteer2':
    for field,value in n['target'].items():ordinal[field][str(value)]+=1
   else:labels[json.dumps(n['target'],sort_keys=True)]+=1
  answer[source]={'counts':dict(counts),'eligible_ids':eligible,'eligible_native_target_counts':dict(labels),'eligible_ordinal_rating_counts':{k:dict(v) for k,v in ordinal.items()},'adapter_review':dict(errors)}
 return {'status':'audited_not_released','v1_native_sha256':sha(native_path),'v1_typed_sha256':sha(typed_path),'parent_manifest_sha256':sha(parent/'manifest.json'),'complete_local_benchmark_pools':pools,'sources':answer,'scope':'Read-only source-membership, whole local benchmark-pool text overlap and parent held-out/context screening. No new split/release, fitting or GPU trial. Explicit source datasets differ from every test benchmark; text matching cannot exclude unknown paraphrases. GSM8K synthetic alternatives require separate validation. Ordinal counts are record bundles, not an established matched-decision condition.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--parent',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Preserve previous audit')
 result=audit(a.root,a.parent);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v['counts'] for k,v in result['sources'].items()},indent=2))
