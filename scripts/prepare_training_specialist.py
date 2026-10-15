import json, hashlib, random, gzip, urllib.request, concurrent.futures
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'configs/training-source-pins.json').read_text()); SEED=20261014
SOURCES=['HelpSteer2','PubMedQA','RuleTaker','ARC','GSM8K']
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def download(task):
 s,rev,f=task; p=ROOT/'data/training/raw'/s/f['path']; p.parent.mkdir(parents=True,exist_ok=True)
 url=f'https://huggingface.co/datasets/{D[s]["repo"]}/resolve/{rev}/{f["path"]}'
 if not p.exists():
  with urllib.request.urlopen(url,timeout=180) as r, p.open('wb') as out:
   while b:=r.read(1024*1024): out.write(b)
 h=digest(p)
 if f.get('lfs') and h!=f['lfs']['oid']: raise ValueError(f'Checksum mismatch {p}')
 return {'path':str(p.relative_to(ROOT)),'url':url,'sha256':h,'bytes':p.stat().st_size}
def files(s):
 if s=='HelpSteer2': return D[s]['native_revision'],[{'path':'train.jsonl.gz'},{'path':'preference/preference.jsonl.gz'}]
 fs=D[s]['files']
 if s=='PubMedQA':fs=[f for f in fs if f['path'].startswith('pqa_artificial/')]
 if s=='GSM8K':fs=[f for f in fs if f['path'].startswith('main/')]
 return D[s]['revision'],fs
if __name__=='__main__':
 tasks=[(s,files(s)[0],f) for s in SOURCES for f in files(s)[1]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
  for task,result in zip(tasks,ex.map(download,tasks)):
   print(task[0],result,flush=True)
 print('DOWNLOAD_COMPLETE',flush=True)
 def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 def normalize(s,r,config,index):
  if s=='RuleTaker': config=r['config']
  identifier=r.get('id',r.get('pubid',index)); group=r.get('prompt',r.get('context',r.get('question',identifier)))
  if s=='HelpSteer2':
   if config=='ratings':
    inp={'state':{'prompt':r['prompt'],'response':r['response']},'question':'Rate the response on each documented dimension.','options':None}
    target={k:r[k] for k in ['helpfulness','correctness','coherence','complexity','verbosity']}; cap={'score':'native 0–4 ordinal dimensions; keep dimension rubrics distinct','choice':'recast each ordinal dimension as ordered choices','noul':'requires explicitly defined threshold'}
   else:
    inp={'state':{'prompt':r['prompt'],'response_1':r['response_1'],'response_2':r['response_2']},'question':'Which response is preferred?','options':{'response_1':'Response 1','tie':'No preference','response_2':'Response 2'}}
    z=r['preference_strength']; target={'choice':'response_2' if z>0 else 'response_1' if z<0 else 'tie','preference_strength':z}; cap={'choice':'native sign recast including tie','score':'native signed preference strength -3 to 3','noul':'recast Response 2 preferred with tie policy explicitly specified'}
  elif s=='PubMedQA':
   inp={'state':r['context']['contexts'],'question':r['question'],'options':{'yes':'Yes','no':'No','maybe':'Maybe'}};target=r['final_decision'];cap={'choice':'native yes/no/maybe','noul':'requires explicit maybe policy','score':None};group=r['pubid']
  elif s=='RuleTaker':
   inp={'state':r['context'],'question':r['question'],'options':{'entailment':'Entailed','contradiction':'Contradicted','neutral':'Unknown'}};target=r['label'];cap={'choice':'native NLI label','noul':'entailed? recast; non-entailment includes neutral','score':None};group=r['context']
  elif s=='ARC':
   inp={'state':'','question':r['question'],'options':dict(zip(r['choices']['label'],r['choices']['text']))};target=r['answerKey'];cap={'choice':'native answer choices','noul':'candidate answer verification recast','score':None};group=r['id']
  else:
   inp={'state':'','question':r['question'],'options':None};target={'numeric_answer':r['answer'].rsplit('####',1)[1].strip()};cap={'choice':'requires generated independently validated distractors','noul':'requires proposed numeric answer to verify','score':'numeric answer is not an ordinal rubric score'};group=r['question']
  return {'id':f'{s}:{config}:{identifier}','source':s,'source_config':config,'source_split':'train','source_index':index,'group_id':f'{s}:{stable(group)}','input':inp,'target':target,'raw_native':r,'typed_capabilities':cap}
 for s in SOURCES:
  rev,fs=files(s); metadata=[];selected=[];reserve=[];pools={};pool_labels={};pool_groups={};rng=random.Random(SEED)
  for f in fs:
   p=ROOT/'data/training/raw'/s/f['path'];metadata.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size,'url':f'https://huggingface.co/datasets/{D[s]["repo"]}/resolve/{rev}/{f["path"]}'})
  if s=='HelpSteer2':
   for f,config in zip(fs,['ratings','preferences']):
    with gzip.open(ROOT/'data/training/raw'/s/f['path'],'rt') as stream: allrows=[(i,json.loads(line)) for i,line in enumerate(stream)]
    rows=[(i,r) for i,r in allrows if r.get('split','train')=='train'];pools[config]=len(rows)
    from collections import Counter
    pool_groups[config]=len(set(r['prompt'] for _,r in rows))
    pool_labels[config]=dict(Counter(str(r['preference_strength']) for _,r in rows)) if config=='preferences' else {k:dict(Counter(str(r[k]) for _,r in rows)) for k in ['helpfulness','correctness','coherence','complexity','verbosity']}
    # Preserve original selected indices: advance shared RNG exactly as original code.
    chosen=rng.sample(rows,125)
    for i,r in chosen:selected.append(normalize(s,r,config,i))
    taken={i for i,_ in chosen}
    remaining=[pair for pair in rows if pair[0] not in taken]
    reserve_rng=random.Random(SEED+1+(0 if config=='ratings' else 1))
    for i,r in reserve_rng.sample(remaining,25):reserve.append(normalize(s,r,config,i))
  else:
   tables=[];offset=0;ranges=[]
   for f in fs:
    table=pq.read_table(ROOT/'data/training/raw'/s/f['path']);config=f['path'].split('/')[0];pools[config]=table.num_rows;ranges.append((offset,offset+table.num_rows,config,table));offset+=table.num_rows
    from collections import Counter
    lc={'PubMedQA':'final_decision','RuleTaker':'label','ARC':'answerKey','GSM8K':'answer'}[s]
    pool_labels[config]=dict(Counter(str(x).rsplit('####',1)[-1].strip() if s=='GSM8K' else str(x) for x in table[lc].to_pylist()))
    gc={'PubMedQA':'pubid','RuleTaker':'context','ARC':'id','GSM8K':'question'}[s]
    pool_groups[config]=len(set(table[gc].to_pylist()))
   chosen_indices=rng.sample(range(offset),250)
   chosen_set=set(chosen_indices)
   reserve_indices=random.Random(SEED+1).sample([i for i in range(offset) if i not in chosen_set],50)
   for idx in chosen_indices+reserve_indices:
    for start,end,config,table in ranges:
     if start<=idx<end:
      i=idx-start;r=table.slice(i,1).to_pylist()[0];(selected if idx in chosen_set else reserve).append(normalize(s,r,config,i));break
  out=ROOT/'data/training/processed'/f'{s}.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('w') as stream:
   for r in selected:stream.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
  reserve_out=ROOT/'data/training/reserve'/f'{s}.jsonl';reserve_out.parent.mkdir(parents=True,exist_ok=True)
  reserve_out.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in reserve))
  from collections import Counter
  labels=Counter(json.dumps(r['target'],sort_keys=True) for r in selected)
  manifest={'source':s,'repository':D[s]['repo'],'revision':rev,'license':D[s]['license'],'sampling_seed':SEED,'sampling_method':'uniform random without replacement within source; HelpSteer2 quota 125 ratings +125 train preferences','pool_counts':pools,'pool_label_counts':pool_labels,'pool_group_counts':pool_groups,'sample_count':len(selected),'reserve_count':len(reserve),'reserve_file':str(reserve_out.relative_to(ROOT)),'reserve_sha256':digest(reserve_out),'reserve_seed':SEED+1,'reserve_preference_seed':SEED+2 if s=='HelpSteer2' else None,'reserve_selected_ids':[r['id'] for r in reserve],'sample_group_count':len(set(r['group_id'] for r in selected)),'selected_ids':[r['id'] for r in selected],'sample_target_counts':dict(labels),'raw_files':metadata,'output_file':str(out.relative_to(ROOT)),'output_sha256':digest(out),'status':'staging_pending_parent_leakage_screen','notes':['Training pools only; native HelpSteer2 preference gzip bundles train and val, and only train records enter the pool.','Raw native labels and rationales retained outside input.','No schema variants or train/calibration/evaluation split created.']}
  (ROOT/'data/training/manifests'/f'{s}.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(s,'PROCESSED',len(selected),pools,flush=True)
