import json, pathlib, urllib.request, hashlib, random, csv, concurrent.futures, collections
import pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parents[1]; D=json.load(open(ROOT/'configs/training-source-pins.json')); SEED=20261014
NAMES=['MultiNLI','SNLI','ANLI','MASSIVE','BANKING77','CivilComments','Aegis2','SST2']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def download(url,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  for attempt in range(4):
   try:
    urllib.request.urlretrieve(url,str(p)+'.tmp'); pathlib.Path(str(p)+'.tmp').rename(p);break
   except Exception:
    if attempt==3:raise
 return p

def options(d,field):
 info=d.get('card',{}).get('dataset_info',{})
 if isinstance(info,list):info=info[0]
 for f in info.get('features',[]):
  if f['name']==field and isinstance(f.get('dtype'),dict):return f['dtype']['class_label']['names']
 return {}
def run(n):
 d=D[n]; files=[]; banking=None
 if n=='BANKING77':
  rev=d['acquisition_revision']
  d=dict(repo='PolyAI-LDN/task-specific-datasets',revision=rev,native_revision=rev,license='cc-by-4.0')
  path='banking_data/train.csv'; url=f'https://raw.githubusercontent.com/{d["repo"]}/{rev}/{path}'; p=download(url,ROOT/'data/training/raw'/n/'train.csv'); files=[dict(path=path,local_path=str(p.relative_to(ROOT)),url=url,sha256=sha(p),bytes=p.stat().st_size)]; banking=list(csv.DictReader(open(p))); total=len(banking)
 else:
  fs=d['files'];fs=[f for f in fs if f['path'].startswith('all_1.1/')] if n=='MASSIVE' else fs
  for f in fs:
   path=f['path']; revision=d['revision']; url=f'https://huggingface.co/datasets/{d["repo"]}/resolve/{revision}/{path}'
   p=download(url,ROOT/'data/training/raw'/n/path); checksum=sha(p)
   if f.get('lfs') and checksum!=f['lfs']['oid']:raise ValueError('Checksum mismatch '+n)
   files.append(dict(path=path,local_path=str(p.relative_to(ROOT)),url=url,sha256=checksum,bytes=p.stat().st_size,rows=pq.ParquetFile(p).metadata.num_rows))
  total=sum(f['rows'] for f in files)
 # Sample uniformly over eligible native training rows, rejecting invalid native NLI labels.
 rng=random.Random(SEED); selected={}; eligible=0; invalid=0
 def rows():
  if banking is not None:
   for i,r in enumerate(banking):yield i,'train',r
  else:
   offset=0
   for f in files:
    pf=pq.ParquetFile(ROOT/f['local_path']);split=pathlib.Path(f['path']).name.split('-')[0] if n=='ANLI' else 'train'
    for batch in pf.iter_batches(batch_size=16384):
     for r in batch.to_pylist():yield offset,split,r;offset+=1
 reservoir=[]
 for i,split,r in rows():
  if n in ['MultiNLI','SNLI','ANLI'] and r['label'] not in [0,1,2]:invalid+=1;continue
  if n=='Aegis2' and (r.get('prompt_label_source')!='human' or r.get('prompt_label') not in ['safe','unsafe']):invalid+=1;continue
  eligible+=1
  if len(reservoir)<300:reservoir.append((i,split,r))
  else:
   j=rng.randrange(eligible)
   if j<300:reservoir[j]=(i,split,r)
 reservoir=reservoir[:250]+sorted(reservoir[250:],key=lambda x:x[0]);out=[];labels=collections.Counter()
 for i,split,r in reservoir:
  label=None;opt={};task='choice';group=None
  if n in ['MultiNLI','SNLI','ANLI']:
   state=f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}";opt=options(d,'label');label=opt[str(r['label'])];question='How does the premise relate to the hypothesis?';group=r.get('promptID') if n=='MultiNLI' else hashlib.sha256(r['premise'].encode()).hexdigest()
  elif n=='MASSIVE':
   state=r['utt'];opt={str(i):v for i,v in enumerate(json.loads(pq.ParquetFile(ROOT/files[0]['local_path']).schema_arrow.metadata[b'huggingface'])['info']['features']['intent']['names'])};label=opt.get(str(r['intent']),r['intent']);question='Which assistant intent does the utterance express?';group=str(r['id'])
  elif n=='BANKING77':
   state=r['text'];label=r['category'];opt={k:k.replace('_',' ') for k in sorted({a['category'] for a in banking})};question='Which banking intent does the request express?'
  elif n=='SST2':state=r['sentence'];opt={'0':'negative','1':'positive'};label=opt[str(r['label'])];question='What is the sentiment of this movie-review text?'
  elif n=='CivilComments':state=r['text'];label=r['toxicity'];task='noul';question='Would an annotator consider this comment toxic?'
  elif n=='Aegis2':
   state=r['prompt'];task='noul';question='Is this user prompt unsafe?';label=(r['prompt_label']=='unsafe');opt={}
  if group is None:group=hashlib.sha256(state.encode()).hexdigest()
  labels[json.dumps(label,sort_keys=True)]+=1
  out.append(dict(id=f'{n}:{split}:{i}',source=n,source_revision=d['revision'],native_revision=d.get('native_revision'),source_split=split,source_row_index=i,group_id=f'{n}:{group}',raw_native=r,state=state,task=task,question=question,options=opt,target=label,staging_status='pending contamination screening'))
 reserve=out[250:];out=out[:250]; labels=collections.Counter(json.dumps(r['target'],sort_keys=True) for r in out)
 (ROOT/'data/training/reserve').mkdir(exist_ok=True)
 (ROOT/'data/training/reserve'/f'{n}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in reserve))
 p=ROOT/'data/training/processed'/f'{n}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in out))
 manifest=dict(source=n,repo=d['repo'],revision=d['revision'],native_revision=d.get('native_revision'),license=d.get('license'),files=files,pool_count=total,eligible_count=eligible,excluded_invalid_label_count=invalid,sample_count=len(out),reserve_count=len(reserve),reserve_path=str((ROOT/'data/training/reserve'/f'{n}.jsonl').relative_to(ROOT)),reserve_sha256=sha(ROOT/'data/training/reserve'/f'{n}.jsonl'),sample_seed=SEED,sampling='uniform reservoir sample of eligible training rows; Python random.Random; no schema variants',sample_group_count=len({r['group_id'] for r in out}),label_distribution=dict(labels),output_sha256=sha(p),output_path=str(p.relative_to(ROOT)),contamination_status='staging; not certified contamination-free',notes=[])
 if n=='SNLI':manifest['notes'].append('Export omits original image IDs; premise hash is surrogate family grouping, not certified image-level grouping.')
 if n=='SST2':manifest['notes'].append('Native export lacks sentence-tree family IDs; exact text hash grouping cannot detect all phrase-parent relationships.')
 if n=='CivilComments':manifest['notes'].append('Toxicity is annotation fraction; no binary threshold applied.')
 if n=='Aegis2':manifest['notes'].append('All native safety labels retained; sample restricted to human prompt labels safe/unsafe; response labels and jury provenance remain in raw_native only; no confidence invented.')
 (ROOT/'data/training/manifests'/f'{n}.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(n,total,len(out),flush=True)
 return n
if __name__=='__main__':
 import sys
 names=sys.argv[1:] or NAMES
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  fut={ex.submit(run,n):n for n in names};failed=[]
  for f in concurrent.futures.as_completed(fut):
   try:f.result()
   except Exception as e:failed.append(fut[f]);print('FAILED',fut[f],repr(e),flush=True)
  if failed:raise SystemExit('Failed sources: '+', '.join(failed))
