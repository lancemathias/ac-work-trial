"""Fail closed on task-source exclusions and normalized full-pool text matches."""
import collections,hashlib,json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'ANLI','MultiRC','RACE','ReClor','JEEBench','HelpSteer2','Implicatures','CICERO','CaseHOLD'}
FORBIDDEN={'winogrande','financialphrasebank','ragtruth','judgebench','bbh','bigbenchhard','jevbench','tabfact','contractnli','circa','belebele','truthfulqa'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from strings(v)
 elif isinstance(x,list):
  for v in x:yield from strings(v)
def normalized(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def audit():
 source=ROOT/'data/training/dataset-v3.jsonl';rows=[json.loads(x) for x in source.open()]
 expected=json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites']
 index=collections.defaultdict(set);receipts=[]
 for name,entry in sorted(expected.items()):
  p=ROOT/'data/processed'/(name+'.jsonl');assert sha(p)==entry['sha256'],name
  n=0
  for line in p.open():
   n+=1;row=json.loads(line)
   for text in strings(row['state']):
    value=normalized(text)
    if len(value.split())>=8:index[hashlib.sha256(value.encode()).hexdigest()].add(name)
  receipts.append({'suite':name,'records':n,'sha256':sha(p)})
 assert len(receipts)==11
 hits=[]
 for r in rows:
  if r['suite'] not in ALLOWED or r.get('evaluation_only'):hits.append({'id':r['id'],'reason':'unapproved source'})
  # Source metadata, rather than task content, determines excluded task membership.
  metadata=re.sub('[^a-z0-9]','',json.dumps(r.get('source',{})).casefold())
  for name in FORBIDDEN:
   if name in metadata:hits.append({'id':r['id'],'reason':'excluded source lineage','source':name})
  for text in strings({'state':r['state'],'question':r['question']}):
   value=normalized(text)
   if len(value.split())<8:continue
   suites=index.get(hashlib.sha256(value.encode()).hexdigest())
   if suites:hits.append({'id':r['id'],'reason':'full-pool normalized exact text overlap','suites':sorted(suites)})
 report={'approved':not hits,'dataset_sha256':sha(source),'records':len(rows),'sources':dict(collections.Counter(r['suite'] for r in rows)),'benchmark_pools':receipts,'hits':hits,'scope':'All v3 records (before train/dev/calibration splitting); all locally materialized eligible benchmark examples, not just sampled panel. Task-source allowlist excludes entire eleven datasets.','lineage':'JEEBench (dair-iitd examination corpus) is distinct from JevBench. Implicatures (George/Mamidi) distinct from Circa; CICERO from DailyDialog/DREAM/MuTual; CaseHOLD judicial holdings distinct from ContractNLI contracts. Prior v3 addition audit includes near/containment screening.','limitations':'Source provenance and text screening cannot prove absence of unknown upstream paraphrases or pretraining exposure.'}
 out=ROOT/'outputs/autoresearch/data-audit.json';out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'approved':report['approved'],'records':len(rows),'pool_records':sum(r['records'] for r in receipts),'hits':hits[:20]}))
 if hits:raise ValueError('Data exclusion audit failed')
 return report
if __name__=='__main__':audit()
