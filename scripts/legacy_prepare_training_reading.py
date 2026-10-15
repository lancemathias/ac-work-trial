import json,hashlib,random,urllib.request,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]; SEED=20261014
D=json.loads((ROOT/'configs/training-source-pins.json').read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def download(url,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  with urllib.request.urlopen(url,timeout=180) as r: p.write_bytes(r.read())
 return {'url':url,'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size}
def prep(k):
 d=D[k]; fs=d['files']
 if k=='RACE': fs=[f for f in fs if f['path'].startswith('all/')]
 if k=='MultiRC': fs=[f for f in fs if f['path'].startswith('multirc/')]
 raw=[]; rows=[]
 for f in fs:
  p=ROOT/'data/training/raw'/k/f['path']; url=f"https://huggingface.co/datasets/{d['repo']}/resolve/{d['revision']}/{f['path']}"
  meta=download(url,p); expected=f.get('lfs',{}).get('oid')
  if expected: assert meta['sha256']==expected,(k,meta['sha256'],expected)
  raw.append(meta)
  for i,r in enumerate(pq.read_table(p).to_pylist()): rows.append((f['path'],i,r))
 return rows,raw,d

def convert(k,file,i,r):
 rid=r.get('example_id') or r.get('metadata_question_id') or r.get('id') or r.get('id_string') or f'{file}:{i}'
 state=''; options=[]; target=None; task='choice'; family=None
 if k=='RACE': state=r['article']; q=r['question']; options=r['options']; target=ord(r['answer'])-65; family=hashlib.sha256(state.encode()).hexdigest()
 elif k=='SciQ': state=r['support']; q=r['question']; options=[r['correct_answer'],r['distractor1'],r['distractor2'],r['distractor3']]; target=0; family=hashlib.sha256(state.encode()).hexdigest() if state else rid
 elif k=='MultiRC': state=r['paragraph']; q=r['question']; task='binary_answer_correctness'; options=[r['answer']]; target=bool(r['label']); family=str(r['idx']['paragraph']); rid=json.dumps(r['idx'],sort_keys=True)
 elif k=='MCTest':
  state=r.get('story',r.get('passage','')).replace('\\newline','\n'); q=r.get('question',''); options=[r['answer_options'][a] for a in 'ABCD']; target=r.get('answer',r.get('label')); family=hashlib.sha256(state.encode()).hexdigest()
 elif k=='ReClor': state=r['context']; q=r['question']; options=r['answers']; target=r['label']; family=hashlib.sha256(state.encode()).hexdigest()
 elif k=='WIQA': state='\n'.join(r['question_para_step']); q=r['question_stem']; options=r['choices']['text']; target=r['choices']['label'].index(r['answer_label_as_choice']); family=r['metadata_para_id']
 elif k=='MCScript2': state=r['text']; q=r['question']; options=[a['text'] for a in r['answers']]; target=next(j for j,a in enumerate(r['answers']) if a['correct']); family=r['document_id']
 if isinstance(options,dict): options=options.get('text',[])
 if k=='MCTest' and isinstance(target,str): target=ord(target)-65 if target in 'ABCD' else int(target)
 return {'id':f'{k}:{rid}','source':k,'split':'train','family_id':f'{k}:{family}','provenance':{'file':file,'row_index':i},'state':state,'question':q,'task':task,'options':options,'target':target,'raw_native':r}
def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--sources',nargs='+',default=['RACE','SciQ','MultiRC','MCTest','ReClor','WIQA','MCScript2']); args=a.parse_args()
 for k in args.sources:
  if k=='MCScript2':
   p=ROOT/'data/training/raw'/k/'MCScript-2.0.zip';raw=[download('https://fedora.clarin-d.uni-saarland.de/sfb1102/MCScript-2.0.zip',p)];d={'repo':'official Saarland release','revision':raw[0]['sha256'],'license':'see upstream release; not inferred'}
   z=zipfile.ZipFile(p); names=[n for n in z.namelist() if n.endswith('train-data.xml')];assert len(names)==1,names
   root=ET.fromstring(z.read(names[0]));rows=[]
   for doc in root.findall('.//instance'):
    for q in doc.findall('.//question'):
     r={'document_id':doc.get('id'),'id':doc.get('id')+':'+q.get('id'),'text':doc.findtext('text'),'question':q.get('text'),'answers':[{'text':x.get('text'),'correct':x.get('correct')=='True','id':x.get('id')} for x in q.findall('.//answer')]};rows.append((names[0],len(rows),r))
  else: rows,raw,d=prep(k)
  # Sample uniformly over native datapoints, not expanded schemas. Sorted indices preserve native ordering.
  draw=random.Random(SEED).sample(range(len(rows)),300); inds=sorted(draw[:250]); reserve=[convert(k,*rows[i]) for i in sorted(draw[250:])]; rp=ROOT/'data/training/reserve'/f'{k}.jsonl'; rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in reserve)); out=[convert(k,*rows[i]) for i in inds]
  dest=ROOT/'data/training/processed'/f'{k}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in out))
  from collections import Counter
  m={'source':k,'status':'staging_pending_overlap_screening','repo':d['repo'],'revision':d['revision'],'license':d.get('license'),'raw_files':raw,'eligible_pool_count':len(rows),'sample_count':250,'sample_seed':SEED,'sampling':'uniform native training rows without replacement; equal source quota','sample_indices':inds,'label_counts':dict(Counter(str(x['target']) for x in out)),'sample_groups':len(set(x['family_id'] for x in out)),'processed_sha256':digest(dest),'conversion':'native labels preserved; no schema expansion, no split assignment'}
  mp=ROOT/'data/training/manifests'/f'{k}.json';mp.parent.mkdir(parents=True,exist_ok=True);mp.write_text(json.dumps(m,indent=2)+'\n');print(k,len(rows),len(out),flush=True)
if __name__=='__main__':main()
