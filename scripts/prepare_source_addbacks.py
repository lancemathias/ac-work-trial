"""Normalized, matched-decision source add-backs with fixed E3 hold-outs."""
import argparse,collections,copy,hashlib,heapq,json,re,shutil,unicodedata
from pathlib import Path
from legacy_generate_typed_training import convert as typed_convert
ROOT=Path(__file__).resolve().parents[1];SEED=20261014
CONDITIONS={'ruletaker':{'RuleTaker':250},'aegis2':{'Aegis2':250},'civilcomments':{'CivilComments':250},'snli-multinli':{'SNLI':125,'MultiNLI':125},'massive':{'MASSIVE':250},'banking77':{'BANKING77':250},'existing-source-control':{'ANLI':63,'RACE':63,'ReClor':62,'MultiRC':62}}
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).open()]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def words(s):return re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold())
def key(s):return hashlib.sha256(' '.join(words(s)).encode()).hexdigest()
def texts(x):
 if isinstance(x,str):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from texts(v)
 elif isinstance(x,list):
  for v in x:yield from texts(v)
def context_keys(state):
 result={key(s) for s in texts(state) if len(words(s))>=8}
 if isinstance(state,str) and state.startswith('Premise: ') and '\nHypothesis: ' in state:result.add(key(state.split('\nHypothesis: ',1)[0][9:]))
 return result
def normalize(native,typed):
 content=json.loads(typed['model_payload']['input']);props=typed['model_payload']['schema']['properties']
 if len(props)!=1:raise ValueError('First round requires one decision per native record')
 field,prop=next(iter(props.items()));kind=typed['field_types'][field];label=typed['gold'][field]
 question={'type':kind,'instructions':prop['description'] if kind=='noul' else content['question']}
 if kind=='choice':
  traces=[t for t in typed['transformation_trace'] if t.get('operation')=='choice_permutation' and t.get('field')==field]
  if len(traces)!=1:raise ValueError('Missing native option permutation')
  descriptions=traces[0]['option_descriptions'];options=native.get('input',native)['options'];values=list(options.values()) if isinstance(options,dict) else options
  if collections.Counter(descriptions.values())!=collections.Counter(str(x) for x in values):raise ValueError('Option semantics changed')
  if set(descriptions)!=set(prop['enum']) or label not in descriptions:raise ValueError('Option IDs or target changed')
  question['criteria']=descriptions
 elif kind!='noul':raise ValueError('Ordinal multi-field add-back belongs to later round')
 return {'id':typed['id'],'suite':typed['source'],'state':content['state'],'question':question,'label':label,'group':typed['family_id'],'source':dict(typed['native_source'],normalization='Native question; option meanings in criteria; no v1 schema-description wrapper'),'evaluation_only':False}
def benchmark_index():
 expected=json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites'];exact=collections.defaultdict(set);anchors=collections.defaultdict(set);sets={};receipts=[]
 def grams(w):return {hashlib.sha256(' '.join(w[i:i+5]).encode()).digest()[:12] for i in range(len(w)-4)}
 for suite,spec in expected.items():
  p=ROOT/'data/processed'/(suite+'.jsonl')
  if sha(p)!=spec['sha256']:raise ValueError('Benchmark pool changed')
  count=0
  for row in read(p):
   count+=1
   for text in texts(row['state']):
    w=words(text)
    if len(w)<8:continue
    k=key(text);exact[k].add(suite)
    if len(w)<20 or k in sets:continue
    ss=grams(w);sets[k]=ss
    for anchor in sorted(ss)[:8]:anchors[anchor].add(k)
  receipts.append({'suite':suite,'records':count,'sha256':sha(p)})
 def hit(row):
  for text in texts({'state':row['state'],'question':row['question']}):
   w=words(text);k=key(text)
   if len(w)>=8 and k in exact:return {'kind':'normalized_exact','suites':sorted(exact[k])}
   if len(w)<20:continue
   ss=grams(w);possible={k for g in ss for k in anchors.get(g,())}
   for k in possible:
    other=sets[k];shared=len(ss&other)
    if shared>=20 and (shared/len(ss|other)>=.85 or shared/min(len(ss),len(other))>=.95):return {'kind':'near_or_containment','suites':sorted(exact[k])}
  return None
 return hit,receipts

def control_pool(source_root,source):
 import pyarrow.parquet as pq
 from legacy_prepare_training_reading import convert as reading_convert
 manifest=json.loads((source_root/f'data/training/manifests/{source}.json').read_text());pins=json.loads((ROOT/'configs/training-source-pins.json').read_text());heap=[];serial=0;receipts=[]
 for f in manifest.get('files',manifest.get('raw_files',[])):
  p=source_root/f.get('local_path',f['path'])
  if sha(p)!=f['sha256']:raise ValueError('Native raw file changed')
  receipts.append({'source':source,'path':str(p.relative_to(source_root)),'sha256':sha(p)})
  local=Path(f['path']).name;split=local.split('-')[0] if source=='ANLI' else 'train'
  for batch in pq.ParquetFile(p).iter_batches(batch_size=8192):
   for i,r in enumerate(batch.to_pylist()):
    if source=='ANLI':
     if r['label'] not in [0,1,2]:serial+=1;continue
     n={'id':f'ANLI:{split}:{serial}','source':source,'source_revision':pins[source]['revision'],'source_split':split,'source_row_index':serial,'group_id':'ANLI:'+hashlib.sha256(r['premise'].encode()).hexdigest(),'state':f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}",'question':'How does the premise relate to the hypothesis?','options':{'0':'entailment','1':'neutral','2':'contradiction'},'target':str(r['label'])}
    else:
     n=reading_convert(source,str(p.relative_to(source_root/'data/training/raw'/source)),serial,r)
    rank=int(hashlib.sha256(canonical([SEED,'fresh-control',source,n['id']]).encode()).hexdigest(),16)
    entry=(-rank,serial,n);serial+=1
    if len(heap)<4096:heapq.heappush(heap,entry)
    else:heapq.heappushpop(heap,entry)
 return [x[2] for x in sorted(heap,key=lambda x:(-x[0],x[1]))],receipts

def prepare(source_root,parent,destination):
 source_root=Path(source_root);parent=Path(parent);destination=Path(destination)
 if destination.exists():raise FileExistsError(destination)
 native_path=source_root/'data/training/sample-5000.jsonl';typed_path=source_root/'data/training/typed-5000.jsonl';old=json.loads((source_root/'data/training/manifests/typed-5000.json').read_text())
 if sha(native_path)!=old['input_sha256'] or sha(typed_path)!=old['output_sha256']:raise ValueError('V1 source files changed')
 native={r['id']:r for r in read(native_path)};typed=read(typed_path);m=json.loads((parent/'manifest.json').read_text());splitrows={}
 for role in ['train','dev','calibration']:
  p=parent/m['splits'][role]['points']['path']
  if sha(p)!=m['splits'][role]['points']['sha256']:raise ValueError('Parent split changed')
  splitrows[role]=read(p)
 parentrows=[r for v in splitrows.values() for r in v];parentids={r['id'] for r in parentrows};parentgroups={r['group'] for r in parentrows};parentcontexts={k for r in parentrows for k in context_keys(r['state'])}
 hit,bench_receipts=benchmark_index();rejections=[];raw_receipts=[];selected={}
 for source in ['RuleTaker','Aegis2','CivilComments','SNLI','MultiNLI','MASSIVE','BANKING77']:
  selected[source]=[]
  candidates=sorted([r for r in typed if r['source']==source],key=lambda r:hashlib.sha256(canonical([SEED,'addback',r['id']]).encode()).hexdigest())
  for t in candidates:
   row=normalize(native[t['id']],t);reason='parent ID/group/state overlap' if row['id'] in parentids or row['group'] in parentgroups or context_keys(row['state'])&parentcontexts else hit(row)
   if reason:rejections.append({'id':row['id'],'source':source,'reason':reason});continue
   selected[source].append(row)
 for source in ['ANLI','RACE','ReClor','MultiRC']:
  selected[source]=[];seen_control_ids=set();pool,raw=control_pool(source_root,source);raw_receipts.extend(raw)
  quota=CONDITIONS['existing-source-control'][source]
  for n in pool:
   row=normalize(n,typed_convert(n))
   if row['id'] in seen_control_ids:
    rejections.append({'id':row['id'],'source':source,'reason':'duplicate native control ID'});continue
   seen_control_ids.add(row['id'])
   reason='parent ID/group/state overlap' if row['id'] in parentids or row['group'] in parentgroups or context_keys(row['state'])&parentcontexts else hit(row)
   if reason:rejections.append({'id':row['id'],'source':source,'reason':reason});continue
   selected[source].append(row)
   if len(selected[source])==quota:break
 for quotas in CONDITIONS.values():
  for source,quota in quotas.items():
   if len(selected[source])<quota:raise ValueError(f'Insufficient screened {source}: {len(selected[source])}/{quota}; no condition launched')
 destination.mkdir(parents=True);conditions={}
 for name,quotas in CONDITIONS.items():
  additions=[r for source,quota in quotas.items() for r in selected[source][:quota]]
  if len(additions)!=250 or len({r['id'] for r in additions})!=250:raise ValueError('Unmatched or duplicate added decisions: '+name)
  out=destination/name;out.mkdir();dataset=out/'dataset.jsonl';allrows=splitrows['train']+additions+splitrows['dev']+splitrows['calibration']
  dataset.write_text(''.join(canonical(r)+'\n' for r in allrows));(out/'additions.jsonl').write_text(''.join(canonical(r)+'\n' for r in additions))
  proof={'approved':True,'dataset_sha256':sha(dataset),'parent_dataset_sha256':m['source_dataset_sha256'],'basis':'Only explicit allowed nonbenchmark source datasets; all additions screened against complete locally materialized benchmark pools and all E3 source IDs/groups/context texts. Fixed held-outs unchanged.','benchmark_pools':bench_receipts,'added_sources':quotas,'rejections':len(rejections),'limitations':'Text matching does not prove absence of unknown paraphrases. Source membership excludes entire benchmark tasks.','source_provenance':'RuleTaker is a publicly generated symbolic-rule corpus, not claimed human-authored. Aegis2 uses native human prompt labels; CivilComments uses original >=0.5 toxicity threshold.'}
  write(out/'hygiene-report.json',proof);manifest=copy.deepcopy(m);manifest['source_dataset_sha256']=sha(dataset);manifest['split_rule']='Exact E3 held-outs and base train; 250 screened training-only source additions';manifest['hygiene']={'approved':True,'report':{'path':'hygiene-report.json','sha256':sha(out/'hygiene-report.json')}}
  for role in ['train','dev','calibration']:
   src=parent/m['splits'][role]['points']['path'];p=out/(role+'.jsonl');p.write_bytes(src.read_bytes()+(''.join(canonical(r)+'\n' for r in additions).encode() if role=='train' else b''));manifest['splits'][role]['points']={'path':p.name,'sha256':sha(p)};manifest['splits'][role]['count']=len(splitrows[role])+(250 if role=='train' else 0)
   if role=='train':manifest['splits'][role]['allowed_sources']=sorted(set(manifest['splits'][role]['allowed_sources'])|set(quotas))
  write(out/'manifest.json',manifest)
  conditions[name]={'records':len(allrows),'train_decisions':len(splitrows['train'])+250,'added_decisions':250,'source_counts':dict(collections.Counter(r['suite'] for r in additions)),'added_label_counts':dict(collections.Counter(str(r['label']) for r in additions)),'added_ids':[r['id'] for r in additions],'dataset_sha256':sha(dataset),'additions_sha256':sha(out/'additions.jsonl'),'manifest_sha256':sha(out/'manifest.json'),'splits':{role:manifest['splits'][role]['points']['sha256'] for role in splitrows}}
 release={'state':'prepared_not_launched','seed':SEED,'backbone':'E3 frozen Qwen3.5-9B','v1_typed_sha256':sha(typed_path),'v1_native_sha256':sha(native_path),'parent_manifest_sha256':sha(parent/'manifest.json'),'conditions':conditions,'benchmark_pools':bench_receipts,'control_raw_files':raw_receipts,'rejections':rejections,'qualification':'250 added decisions and identical optimizer steps per condition; token counts/runtime differ. SNLI+MultiNLI uses 125 each. All held-out sets remain fixed. PubMedQA v1 229 Yes/21 No/0 Maybe; RAGTruth collapsed before these removals.'}
 write(destination/'release.json',release);return release
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--parent',required=True);p.add_argument('--destination',required=True);a=p.parse_args();r=prepare(a.source_root,a.parent,a.destination);print(json.dumps({'state':r['state'],'conditions':{k:{x:v[x] for x in ['train_decisions','added_decisions','source_counts']} for k,v in r['conditions'].items()},'rejections':len(r['rejections'])},indent=2))
