"""Prepare later source conditions only; never launches or selects models."""
import argparse,collections,copy,hashlib,json
from pathlib import Path
from prepare_source_addbacks import read,sha,canonical,normalize,write,benchmark_index,context_keys,SEED
CONDITIONS={'sst2':{'SST2':250},'pubmedqa':{'PubMedQA':250},'narrative':{'MCScript2':125,'MCTest':125},'wiqa':{'WIQA':250},'helpsteer2-ordinal-flat':{'HelpSteer2':50}}
def prepare(root,parent,audit_path,destination):
 root=Path(root);parent=Path(parent);out=Path(destination);audit=json.loads(Path(audit_path).read_text())
 if out.exists():raise FileExistsError('Use a fresh destination')
 if audit['parent_manifest_sha256']!=sha(parent/'manifest.json'):raise ValueError('Parent differs from availability audit')
 native_path=root/'data/training/sample-5000.jsonl';typed_path=root/'data/training/typed-5000.jsonl'
 if sha(native_path)!=audit['v1_native_sha256'] or sha(typed_path)!=audit['v1_typed_sha256']:raise ValueError('Pinned source bytes changed')
 native={r['id']:r for r in read(native_path)};typed={r['id']:r for r in read(typed_path)}
 pm=json.loads((parent/'manifest.json').read_text());splits={}
 for role in ['train','dev','calibration']:
  p=parent/pm['splits'][role]['points']['path']
  if sha(p)!=pm['splits'][role]['points']['sha256']:raise ValueError('Parent split changed')
  splits[role]=read(p)
 parentrows=[r for values in splits.values() for r in values];parentids={r['id'] for r in parentrows};parentgroups={r['group'] for r in parentrows};parentcontexts={k for r in parentrows for k in context_keys(r['state'])}
 hit,pools=benchmark_index()
 if pools!=audit['complete_local_benchmark_pools']:raise ValueError('Benchmark screening pool changed')
 out.mkdir(parents=True);receipts={};union=[]
 for name,quotas in CONDITIONS.items():
  additions=[];parents=[]
  for source,count in quotas.items():
   allowed=audit['sources'][source]['eligible_ids']
   chosen=sorted(allowed,key=lambda rid:hashlib.sha256(canonical([SEED,'phase2',source,rid]).encode()).hexdigest())[:count]
   if len(chosen)!=count:raise ValueError('Insufficient screened source: '+source)
   for rid in chosen:
    t=typed[rid];n=native[rid];parents.append(rid)
    if source=='HelpSteer2':
     content=json.loads(t['model_payload']['input']);props=t['model_payload']['schema']['properties'];rows=[]
     for field,prop in sorted(props.items()):
      label=int(t['gold'][field])
      if label!=n['target'][field] or prop['enum']!=['0','1','2','3','4']:raise ValueError('Ordinal native target changed')
      rows.append({'id':rid+':ordinal:'+field,'suite':source,'state':content['state'],'question':{'type':'score','instructions':prop['description'],'criteria':prop['enum']},'label':label,'group':t['family_id'],'source':dict(t['native_source'],native_record_id=rid,ordinal_axis=field,normalization='Single native ordinal axis; current one-question format; all5 axes share parent group'),'evaluation_only':False})
     if len(rows)!=5:raise ValueError('Expected5 native rating axes')
    else:rows=[normalize(n,t)]
    for row in rows:
     if rid in parentids or row['group'] in parentgroups or context_keys(row['state'])&parentcontexts or hit(row):raise ValueError('Selected row fails parent/benchmark isolation')
     additions.append(row)
  if len(additions)!=250 or len({r['id'] for r in additions})!=250:raise ValueError('Unmatched decisions')
  folder=out/name;folder.mkdir();allrows=splits['train']+additions+splits['dev']+splits['calibration'];dataset=folder/'dataset.jsonl'
  dataset.write_text(''.join(canonical(r)+'\n' for r in allrows));(folder/'additions.jsonl').write_text(''.join(canonical(r)+'\n' for r in additions))
  proof={'approved':True,'dataset_sha256':sha(dataset),'parent_dataset_sha256':pm['source_dataset_sha256'],'benchmark_pools':pools,'added_sources':dict(collections.Counter(r['suite'] for r in additions)),'basis':'Explicit nonbenchmark source datasets, checked native-source bytes and normalization, complete local benchmark pools, parent IDs/groups/contexts, unchanged held-out bytes.','limitations':'Unknown paraphrases not excluded; ordinal axes share50 parent records and are intentionally flattened, not a joint-request ablation.'}
  write(folder/'hygiene-report.json',proof);manifest=copy.deepcopy(pm);manifest['source_dataset_sha256']=sha(dataset);manifest['split_rule']='Exact E3 parent plus250 screened decisions; held-outs unchanged';manifest['hygiene']={'approved':True,'report':{'path':'hygiene-report.json','sha256':sha(folder/'hygiene-report.json')}}
  for role in splits:
   source=parent/pm['splits'][role]['points']['path'];p=folder/(role+'.jsonl');p.write_bytes(source.read_bytes()+(''.join(canonical(r)+'\n' for r in additions).encode() if role=='train' else b''));manifest['splits'][role]['points']={'path':p.name,'sha256':sha(p)};manifest['splits'][role]['count']=len(splits[role])+(250 if role=='train' else 0)
   if role=='train':manifest['splits'][role]['allowed_sources']=sorted(set(manifest['splits'][role]['allowed_sources'])|set(quotas))
  write(folder/'manifest.json',manifest)
  receipts[name]={'records':len(allrows),'train_decisions':6512,'added_decisions':250,'source_counts':proof['added_sources'],'added_type_counts':dict(collections.Counter(r['question']['type'] for r in additions)),'selected_native_record_ids':parents,'native_parent_records':len(parents),'unique_added_groups':len({r['group'] for r in additions}),'dataset_sha256':sha(dataset),'additions_sha256':sha(folder/'additions.jsonl'),'manifest_sha256':sha(folder/'manifest.json')};union.extend(additions)
 union_path=out/'union-additions.jsonl';union_path.write_text(''.join(canonical(r)+'\n' for r in union))
 release={'state':'prepared_not_queued','seed':SEED,'parent_manifest_sha256':sha(parent/'manifest.json'),'availability_audit_sha256':sha(audit_path),'generator_sha256':sha(__file__),'conditions':receipts,'union_sha256':sha(union_path),'union_records':len(union),'control':'First-round existing-source-control, same runtime/recipe; not a new fitted baseline','source_quota_rule':'Label-blind SHA256 order. All rows from selected ordinal record retained, one axis per normalized request.','scope':'Later candidate data only; first-round results must inform prioritization before GPU launch.'}
 write(out/'release.json',release);return release
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--parent',required=True);p.add_argument('--audit',required=True);p.add_argument('--destination',required=True);a=p.parse_args();r=prepare(a.root,a.parent,a.audit,a.destination);print(json.dumps({'state':r['state'],'conditions':list(r['conditions']),'union_records':r['union_records']},indent=2))
