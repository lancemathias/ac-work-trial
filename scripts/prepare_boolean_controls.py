"""Type/label-matched, train-only MultiRC controls. Never launches experiments."""
import argparse,collections,copy,hashlib,json
from pathlib import Path
from prepare_source_addbacks import SEED,read,sha,canonical,normalize,write,benchmark_index,context_keys,control_pool,typed_convert

def prepare(root,parent,first_release,destination):
 root,parent,out=Path(root),Path(parent),Path(destination)
 if out.exists():raise FileExistsError('Preserve existing data; use a fresh destination')
 first=json.loads(Path(first_release).read_text());pm=json.loads((parent/'manifest.json').read_text())
 if sha(parent/'manifest.json')!=first['parent_manifest_sha256']:raise ValueError('Different parent manifest')
 splits={}
 for role in ['train','dev','calibration']:
  p=parent/pm['splits'][role]['points']['path']
  if sha(p)!=pm['splits'][role]['points']['sha256']:raise ValueError('Parent partition changed')
  splits[role]=read(p)
 rows=[r for v in splits.values() for r in v];ids={r['id'] for r in rows};groups={r['group'] for r in rows};contexts={k for r in rows for k in context_keys(r['state'])}
 hit,pools=benchmark_index();native,raw=control_pool(root,'MultiRC');eligible={True:[],False:[]};seen=set();rejections=collections.Counter()
 for n in native:
  r=normalize(n,typed_convert(n))
  if r['id'] in seen:rejections['duplicate native ID']+=1;continue
  seen.add(r['id'])
  if r['id'] in ids or r['group'] in groups or context_keys(r['state'])&contexts:rejections['parent ID/group/context']+=1;continue
  if hit(r):rejections['benchmark text overlap']+=1;continue
  if r['question']['type']!='noul' or type(r['label']) is not bool:raise ValueError('Require native Boolean decision')
  eligible[r['label']].append(r)
 for label in eligible:eligible[label].sort(key=lambda r:hashlib.sha256(canonical([SEED,'boolean-control',r['id']]).encode()).hexdigest())
 out.mkdir(parents=True);conditions={};union={}
 for original in ['ruletaker','aegis2','civilcomments']:
  quota=first['conditions'][original]['added_label_counts'];added=[]
  for label in [True,False]:
   count=quota.get(str(label),0)
   if len(eligible[label])<count:raise ValueError('Insufficient screened native class; no substitution or relaxed isolation')
   added.extend(eligible[label][:count])
  if len(added)!=250 or len({r['id'] for r in added})!=250:raise ValueError('Unmatched decision count')
  name='multirc-for-'+original;folder=out/name;folder.mkdir();dataset=folder/'dataset.jsonl';allrows=splits['train']+added+splits['dev']+splits['calibration']
  dataset.write_text(''.join(canonical(r)+'\n' for r in allrows));(folder/'additions.jsonl').write_text(''.join(canonical(r)+'\n' for r in added))
  proof={'approved':True,'dataset_sha256':sha(dataset),'parent_dataset_sha256':pm['source_dataset_sha256'],'benchmark_pools':pools,'added_sources':{'MultiRC':250},'basis':'Native pinned MultiRC only; complete local benchmark pool screening; all parent ID/group/context exclusions; identical held-outs. Match first-round source type and native Boolean class counts.','limitations':'Unknown paraphrases, domain, length and group-multiplicity differences remain. Shared training records across conditions are explicit; no record enters held-outs.'}
  write(folder/'hygiene-report.json',proof);manifest=copy.deepcopy(pm);manifest['source_dataset_sha256']=sha(dataset);manifest['split_rule']='Exact v3 split plus250 train-only type/label-matched MultiRC decisions';manifest['hygiene']={'approved':True,'report':{'path':'hygiene-report.json','sha256':sha(folder/'hygiene-report.json')}}
  for role in splits:
   p=folder/(role+'.jsonl');p.write_bytes((parent/pm['splits'][role]['points']['path']).read_bytes()+(''.join(canonical(r)+'\n' for r in added).encode() if role=='train' else b''));manifest['splits'][role]['points']={'path':p.name,'sha256':sha(p)};manifest['splits'][role]['count']=len(splits[role])+(250 if role=='train' else 0)
  write(folder/'manifest.json',manifest)
  conditions[name]={'records':len(allrows),'train_decisions':6512,'added_decisions':250,'source_counts':{'MultiRC':250},'added_type_counts':{'noul':250},'added_label_counts':dict(collections.Counter(str(r['label']) for r in added)),'matched_original_condition':original,'unique_added_groups':len({r['group'] for r in added}),'dataset_sha256':sha(dataset),'additions_sha256':sha(folder/'additions.jsonl'),'manifest_sha256':sha(folder/'manifest.json')}
  if conditions[name]['added_label_counts']!=quota:raise ValueError('Native class counts differ')
  union.update({r['id']:r for r in added})
 p=out/'union-additions.jsonl';p.write_text(''.join(canonical(union[k])+'\n' for k in sorted(union)))
 release={'state':'prepared_not_queued','seed':SEED,'parent_manifest_sha256':sha(parent/'manifest.json'),'first_release_sha256':sha(first_release),'generator_sha256':sha(__file__),'conditions':conditions,'union_sha256':sha(p),'union_records':len(union),'raw_files':raw,'eligible_native_label_counts':{str(k):len(v) for k,v in eligible.items()},'rejections':dict(rejections),'benchmark_pools':pools,'scope':'Conditional follow-up controls; first-grid source outcomes determine whether to launch. Shared native training rows across conditions reduce cache size and are not independent replicate data.'}
 write(out/'release.json',release);return release

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--parent',required=True);p.add_argument('--first-release',required=True);p.add_argument('--destination',required=True);a=p.parse_args();r=prepare(a.root,a.parent,a.first_release,a.destination);print(json.dumps({'conditions':r['conditions'],'union_records':r['union_records'],'eligible_native_label_counts':r['eligible_native_label_counts']},indent=2))
