"""Add screened, group-deduplicated Tasksource examples to frozen v2.1."""
import collections,hashlib,json,pathlib,re,unicodedata
import pyarrow.parquet as pq
from jsonschema import Draft202012Validator
from benchmark_training_data import adapt_training_point
ROOT=pathlib.Path(__file__).resolve().parents[1];AUD=ROOT/'outputs/tasksource-source-audit';SEED=20261022
SOURCES={'implicatures':'Implicatures','cicero':'CICERO','lex_glue/case_hold':'CaseHOLD'}
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def h(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from strings(v)
 elif isinstance(x,list):
  for v in x:yield from strings(v)
def words(s):return re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold())
def grams(w,n):return {hashlib.sha256(' '.join(w[i:i+n]).encode()).digest()[:12] for i in range(len(w)-n+1)}
def main():
 rev=json.load(open(AUD/'revision.json'))['revision'];tree={r['path']:r for r in json.load(open(AUD/'filtered-full-pinned-tree.json')) if r['path'].endswith('.parquet')};candidates=collections.defaultdict(list);reasons=collections.Counter();raw=[]
 for p in sorted((AUD/'filtered-parquet').glob('*.parquet')):
  key='filtered-full/'+p.name;assert sha(p)==tree[key]['lfs']['oid'];raw.append({'file':key,'sha256':sha(p)})
  for b in pq.ParquetFile(p).iter_batches(batch_size=8192):
   for r in b.to_pylist():
    if r['source'] not in SOURCES:continue
    if r['split']!='train' or r['kind']!='choice':reasons[(r['source'],'non_train_or_nonchoice')]+=1;continue
    if len(r['target'])!=len(r['options']) or sum(z==1 for z in r['target'])!=1 or any(z not in [0,1] for z in r['target']):reasons[(r['source'],'not_hard_onehot')]+=1;continue
    if len(r['options'])<2 or len(set(r['options']))!=len(r['options']):reasons[(r['source'],'degenerate_options')]+=1;continue
    candidates[r['source']].append(r)
 # Full exact-text index plus approximate near/containment index against all eleven benchmarks.
 exact=collections.defaultdict(set);anchors=collections.defaultdict(set);sets={};bench_receipts=[]
 expected=json.load(open(ROOT/'data/manifests/benchmark-preprocessing.json'))['suites']
 for p in sorted(pathlib.Path('/Users/bwt/ac-worktrial/data/processed').glob('*.jsonl')):
  assert sha(p)==expected[p.stem]['sha256'];bench_receipts.append({'file':p.name,'sha256':sha(p)})
  for l in p.open():
   row=json.loads(l)
   for text in strings(row['state']):
    w=words(text)
    if len(w)<8:continue
    key=h(w);exact[key].add(p.stem)
    if len(w)<20 or key in sets:continue
    ss=grams(w,5);sets[key]=ss
    for anchor in sorted(ss)[:8]:anchors[anchor].add(key)
 def matches(text):
  w=words(text);key=h(w)
  if len(w)>=8 and key in exact:return {'kind':'normalized_exact','suites':sorted(exact[key])}
  if len(w)<20:return None
  ss=grams(w,5);possible=set(e for g in ss for e in anchors.get(g,()))
  for e in sorted(possible):
   other=sets[e];shared=len(ss&other)
   if shared<20:continue
   j=shared/len(ss|other);c=shared/min(len(ss),len(other))
   if j>=.85 or c>=.95:return {'kind':'5word_near_or_containment','suites':sorted(exact[e]),'jaccard':j,'containment':c}
  return None
 additions=[];selection={};reject=[];dedup_states=set();schema=Draft202012Validator(json.load(open(ROOT/'configs/normalized-training.schema.json')))
 for source,name in SOURCES.items():
  # One decision per source group AND exact state; prefer original direct rendering.
  group_rows={}
  for r in sorted(candidates[source],key=lambda r:(r['variant']!='direct',r['id'])):
   group_rows.setdefault(r['group_id'],r)
  chosen=[]
  for r in sorted(group_rows.values(),key=lambda r:h([SEED,source,r['group_id']])):
   statehash=h(words(r['state']))
   if statehash in dedup_states:reasons[(source,'duplicate_state')]+=1;continue
   hit=matches(r['state']) or matches(r['question'])
   if hit:reject.append({'id':r['id'],'source':source,'hit':hit});continue
   order=sorted(range(len(r['options'])),key=lambda i:h([SEED,r['id'],'option',i]));criteria={f'option_{j}':r['options'][i] for j,i in enumerate(order)};correct=r['target'].index(1);label=f'option_{order.index(correct)}'
   row={'id':'Tasksource:'+r['id'],'suite':name,'state':r['state'],'question':{'type':'choice','instructions':r['question'],'criteria':criteria},'label':label,'group':'Tasksource:'+r['group_id'],'source':{'tasksource_source':source,'tasksource_revision':rev,'native_id':r['id'],'variant':r['variant'],'split':'train','license':r['license']},'evaluation_only':False}
   adapt_training_point(row,schema);chosen.append(row);dedup_states.add(statehash)
   if len(chosen)==1000:break
  assert chosen,(source,'no eligible rows');additions.extend(chosen);selection[source]={'eligible_choice_rows':len(candidates[source]),'source_groups':len(group_rows),'selected':len(chosen),'ids':[r['id'] for r in chosen]}
 basepath=ROOT/'data/training/train-5053-six-sources-quality-preference.jsonl';base=[json.loads(l) for l in basepath.open()];rows=base+additions;assert len({r['id'] for r in rows})==len(rows)
 target=ROOT/'data/training/dataset-v3.jsonl';new=ROOT/'data/training/dataset-v3-additions.jsonl'
 for path,records in [(target,rows),(new,additions)]:path.write_text(''.join(canonical(r)+'\n' for r in records))
 assert rows[:5053]==base
 report={'version':'v3','parent_version':'v2.1','base_sha256':sha(basepath),'records':len(rows),'added_records':len(additions),'dataset_sha256':sha(target),'additions_sha256':sha(new),'source_counts':dict(collections.Counter(r['suite'] for r in rows)),'selection':selection,'seed':SEED,'sampling':'SHA256-ranked groups, prefer direct variant, at most one example per source group and normalized state; deterministic option permutation. No replacement or duplicated padding to quota.','rejections':reject,'other_exclusions':{s+':'+reason:n for (s,reason),n in reasons.items()},'benchmark_files':bench_receipts,'screen':'NFKC/casefold exact >=8 words; five-word-shingle Jaccard >=0.85 or containment >=0.95 with >=20 shared distinct shingles; bottom-eight indexed anchors. All 11 local benchmark state text fields protected.','source_lineage':'Implicatures: George & Mamidi 2020 dialogue corpus with generated distractors. CICERO: DailyDialog/DREAM/MuTual. CaseHOLD: Harvard caselaw judicial holdings. None is an identified direct source family of our eleven; broad independent reuse or paraphrases remain possible. Not a contamination-free guarantee.','tasksource_revision':rev,'raw_files':raw,'parent_records_preserved':5053,'request_schema':'normalized-benchmark-v1','cache':'Reuse v2.1 states; new additions require Qwen3.5-9B batch-four extraction.'}
 out=ROOT/'outputs/dataset-v3-audit.json';out.write_text(json.dumps(report,indent=2)+'\n')
 config={'dataset_version':'v3','dataset_path':str(target.relative_to(ROOT)),'dataset_sha256':sha(target),'records':len(rows),'selected_sources':sorted(report['source_counts']),'format':'normalized-benchmark-v1','adapter':'scripts/benchmark_training_data.py','request_wrapper':'run_baselines.payload: questions.q','release_manifest':'data/training/manifests/dataset-v3.json','derivation_script':'scripts/build_dataset_v3.py','audit':str(out.relative_to(ROOT))}
 (ROOT/'configs/active-training-dataset.json').write_text(json.dumps(config,indent=2)+'\n');(ROOT/'data/training/manifests/dataset-v3.json').write_text(json.dumps(dict(config,parent_version='v2.1',source_counts=report['source_counts'],seed=SEED,additions_sha256=sha(new)),indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['records','added_records','source_counts','dataset_sha256','additions_sha256','other_exclusions']},indent=2),flush=True)
 print('overlap_rejections',len(reject),flush=True)
if __name__=='__main__':main()
