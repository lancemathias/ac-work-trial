"""Six-source training set: five selected sources plus score-derived HelpSteer2 pairs."""
import collections,gzip,hashlib,json,re
from pathlib import Path
from finalize_training_sample import texts,norm,shingles
ROOT=Path(__file__).resolve().parents[1];SEED=20261019
DIMS=['helpfulness','correctness','coherence']
BASE={'ANLI','RACE','ReClor','MultiRC','JEEBench'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main(quota=250):
 raw=ROOT/'data/training/raw/HelpSteer2';rating_path=raw/'train.jsonl.gz';pref_path=raw/'preference/preference.jsonl.gz'
 ratings={}
 with gzip.open(rating_path,'rt') as stream:
  for l in stream:
   r=json.loads(l);key=(r['prompt'],r['response']);v={k:r[k] for k in DIMS}
   if key in ratings and ratings[key]!=v:raise ValueError('Conflicting ratings for same prompt/response')
   ratings[key]=v
 pool=[];excluded=collections.Counter()
 with gzip.open(pref_path,'rt') as stream:
  for i,l in enumerate(stream):
   r=json.loads(l)
   if r['split']!='train':excluded['non_train']+=1;continue
   a=ratings.get((r['prompt'],r['response_1']));b=ratings.get((r['prompt'],r['response_2']))
   if a is None or b is None:excluded['missing_rating_join']+=1;continue
   sa,sb=sum(a.values()),sum(b.values())
   if sa==sb:excluded['equal_total']+=1;continue
   identity=stable([r['prompt'],*sorted([r['response_1'],r['response_2']])]);rid='HelpSteer2:score-preference:'+identity
   swap=int(stable([SEED,rid,'swap']),16)%2
   responses=[r['response_1'],r['response_2']];winner=0 if sa>sb else 1
   if swap:responses.reverse();winner=1-winner
   row={'id':rid,'suite':'HelpSteer2','state':{'question':r['prompt'],'response_A':responses[0],'response_B':responses[1]},'question':{'type':'choice','instructions':'Which response is better overall?','criteria':{'A':'Response A','B':'Response B'}},'label':'AB'[winner],'group':'HelpSteer2:'+stable(r['prompt']),'source':{'split':'train','preference_row':i,'label_rule':'higher total of helpfulness, correctness and coherence; ties excluded'},'evaluation_only':False}
   pool.append((stable([SEED,rid,'sample']),row,{'id':rid,'ratings_response_1':a,'ratings_response_2':b,'total_response_1':sa,'total_response_2':sb,'swapped':bool(swap),'native_preference_strength':r['preference_strength']}))
 # Use the same local overlap screen as the previous dataset preparation.
 idx=collections.defaultdict(set);exact=set();lengths={};receipts=[]
 expected=json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites']
 for p in sorted(Path('/Users/bwt/ac-worktrial/data/processed').glob('*.jsonl')):
  assert sha(p)==expected[p.stem]['sha256'];receipts.append({'file':p.name,'sha256':sha(p)})
  for l in p.open():
   for s in texts(json.loads(l)['state']):
    w=norm(s)
    if len(w)<8:continue
    h=stable(w);exact.add(h)
    if h in lengths:continue
    ss=shingles(w);lengths[h]=len(ss)
    for sh in ss:idx[sh].add(h)
 def hit(row):
  for s in texts(row['state']):
   w=norm(s)
   if len(w)<8:continue
   if stable(w) in exact:return True
   ss=shingles(w)
   if len(ss)<5:continue
   counts=collections.Counter(e for sh in ss for e in idx.get(sh,()))
   if any(n>=5 and n/min(len(ss),lengths[e])>=.8 for e,n in counts.items()):return True
  return False
 selected=[];metadata=[];seen=set()
 for _,row,audit in sorted(pool,key=lambda x:x[0]):
  if row['id'] in seen:excluded['duplicate_pair']+=1;continue
  seen.add(row['id'])
  if hit(row):excluded['benchmark_overlap']+=1;continue
  selected.append(row);metadata.append(audit)
  if len(selected)==quota:break
 assert len(selected)==quota
 basepath=ROOT/'data/training/train-4303-benchmark-format.jsonl';base=[json.loads(l) for l in basepath.open() if json.loads(l)['suite'] in BASE];assert len(base)==1053
 rows=base+selected;assert len({r['id'] for r in rows})==1053+quota
 target=ROOT/f'data/training/train-{1053+quota}-six-sources-quality-preference.jsonl'
 target.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
 audit={'records':1053+quota,'sources':dict(collections.Counter(r['suite'] for r in rows)),'question_types':dict(collections.Counter(r['question']['type'] for r in rows)),'dataset_sha256':sha(target),'base_sha256':sha(basepath),'raw_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [rating_path,pref_path]},'seed':SEED,'sampling':'Ascending SHA256([seed,pair_id,sample]); first quota eligible unique pairs passing screen; swap by SHA256([seed,pair_id,swap]) parity. No PRNG.','label_rule':'Sum helpfulness, correctness and coherence with equal weight; higher total wins; equal totals excluded. Native preference label retained only in audit.','dimensions':DIMS,'caveat':'Complexity and verbosity are excluded. Equal-weight quality sum is a derived preference proxy, not the native human pairwise preference.','eligible_pairs_before_screen':len(pool),'exclusions':dict(excluded),'selected_labels':dict(collections.Counter(r['label'] for r in selected)),'evaluation_files':receipts,'screen':'Exact normalized text >=8 words and >=80% 16-word-shingle containment with >=5 shared shingles, all state strings. Not a guarantee against paraphrases or shared ancestry.','selected_pair_metadata':metadata,'cache':'Current cache run targets earlier 4303 source pool. New HelpSteer2 pair requests need additional forward extraction; existing single-rating caches are incompatible.'}
 (ROOT/'outputs/helpsteer2-pairwise-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 c={'dataset_path':str(target.relative_to(ROOT)),'dataset_sha256':sha(target),'records':1053+quota,'selected_sources':sorted(BASE|{'HelpSteer2'}),'format':'normalized-benchmark-v1','request_wrapper':'run_baselines.payload: questions.q','adapter':'scripts/benchmark_training_data.py','derivation_script':'scripts/add_helpsteer2_pairwise.py','audit':'outputs/helpsteer2-pairwise-audit.json'}
 (ROOT/'configs/active-training-dataset.json').write_text(json.dumps(c,indent=2)+'\n')
 print(json.dumps({k:audit[k] for k in ['records','sources','question_types','eligible_pairs_before_screen','exclusions','selected_labels','dataset_sha256']},indent=2))
if __name__=='__main__':main()
