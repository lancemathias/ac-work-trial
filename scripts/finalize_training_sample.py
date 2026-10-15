"""Validate balanced candidate sample and screen text against local evaluation records."""
import argparse, collections, hashlib, json, pathlib, re
ROOT=pathlib.Path(__file__).resolve().parents[1]
ALLOWED={'MultiNLI','SNLI','ANLI','MASSIVE','BANKING77','CivilComments','Aegis2','SST2','RACE','SciQ','MultiRC','MCTest','MCScript2','ReClor','WIQA','HelpSteer2','PubMedQA','RuleTaker','ARC','GSM8K'}
def texts(x):
 if isinstance(x,str): yield x
 elif isinstance(x,dict):
  for v in x.values(): yield from texts(v)
 elif isinstance(x,list):
  for v in x: yield from texts(v)
def norm(s): return re.findall(r'\w+',s.casefold())
def shingles(w): return {hashlib.sha256(' '.join(w[i:i+16]).encode()).digest()[:12] for i in range(len(w)-15)}
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evaluation-dir',type=pathlib.Path,required=True);a=ap.parse_args()
 index=collections.defaultdict(set); exact=collections.defaultdict(set); lengths={}; receipts=[]
 expected=json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites']
 assert {p.stem for p in a.evaluation_dir.glob('*.jsonl')}==set(expected), 'Evaluation suites differ from benchmark manifest'
 for p in sorted(a.evaluation_dir.glob('*.jsonl')):
  filehash=hashlib.sha256(p.read_bytes()).hexdigest()
  assert filehash==expected[p.stem]['sha256'], f'Evaluation checksum mismatch: {p.name}'
  receipts.append({'file':p.name,'sha256':filehash})
  for line in p.open():
   r=json.loads(line)
   for s in texts(r.get('state')):
    w=norm(s)
    if len(w)<8:continue
    h=hashlib.sha256(' '.join(w).encode()).hexdigest();exact[h].add(p.stem)
    if h in lengths:continue
    ss=shingles(w);lengths[h]=len(ss)
    for sh in ss:index[sh].add(h)
 def hits(r):
  found=[]
  payload=r.get('input',{'state':r.get('state'),'question':r.get('question'),'options':r.get('options')})
  for s in texts(payload):
   w=norm(s)
   if len(w)<8:continue
   h=hashlib.sha256(' '.join(w).encode()).hexdigest()
   if h in exact:found.append({'kind':'exact','suites':sorted(exact[h])});continue
   ss=shingles(w)
   if len(ss)<5:continue
   counts=collections.Counter(e for sh in ss for e in index.get(sh,()))
   for e,n in counts.items():
    if n/min(len(ss),lengths[e])>=.8 and n>=5:
     found.append({'kind':'16_word_shingle_containment','suites':sorted(exact[e]),'overlap':n/min(len(ss),lengths[e])})
  return found
 base=ROOT/'data/training';paths=sorted((base/'processed').glob('*.jsonl'));assert {p.stem for p in paths}==ALLOWED
 report={'seed':20261014,'quota_per_source':250,'evaluation_files':receipts,'screen':'Exact normalized fields >=8 words; >=80% containment of 16-word shingles with >=5 shared shingles. State fields only in evaluation; input fields only in candidates. Heuristic, not ancestry clearance.','sources':{},'rejections':[]}
 combined=[]
 for p in paths:
  rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==250
  reserve=[json.loads(x) for x in (base/'reserve'/p.name).read_text().splitlines()]
  keep=[]
  for r in rows:
   match=hits(r)
   if match:report['rejections'].append({'id':r['id'],'hits':match})
   else:keep.append(r)
  for r in reserve:
   if len(keep)==250:break
   match=hits(r)
   if not match:keep.append(r)
  assert len(keep)==250,(p.stem,len(keep))
  for r in keep:
   assert r['source']==p.stem and r.get('target') is not None
   r['screening_status']='passed local heuristic overlap screen; upstream contamination remains uncertain'
  out=base/'screened'/p.name;out.parent.mkdir(exist_ok=True);out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in keep))
  report['sources'][p.stem]={'count':len(keep),'replacements':250-sum(not hits(r) for r in rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'ids':[r['id'] for r in keep]};combined.extend(keep)
 assert len(combined)==5000 and len({r['id'] for r in combined})==5000
 out=base/'sample-5000.jsonl';out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined));report['combined_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();report['total']=len(combined)
 dump(base/'manifests/sample-5000.json',report);print(json.dumps({'total':5000,'rejections':len(report['rejections']),'sha256':report['combined_sha256']}))
if __name__=='__main__':main()
