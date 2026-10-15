"""Add 500 disjoint, screened records per non-JEE source; preserve existing rows."""
import collections,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from prepare_training_reading import convert as native_reading
from generate_typed_training import convert as typed,canonical
from training_data import adapt_training_point
from jsonschema import Draft202012Validator
from finalize_training_sample import texts,norm,shingles
ROOT=Path(__file__).resolve().parents[1];SEED=20261020

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def main():
 original=ROOT/'data/training/train-2553-six-sources-quality-preference.jsonl';old=[json.loads(l) for l in original.open()];oldids={r['id'] for r in old}
 # Same SHA-ranked HelpSteer pool: quota expansion retains the first 250.
 import add_helpsteer2_pairwise
 add_helpsteer2_pairwise.main(quota=1000)
 hs=[json.loads(l) for l in (ROOT/'data/training/train-2053-six-sources-quality-preference.jsonl').open() if json.loads(l)['suite']=='HelpSteer2']
 assert {r['id'] for r in old if r['suite']=='HelpSteer2'} <= {r['id'] for r in hs}
 idx=collections.defaultdict(set);exact=set();lengths={};receipts=[]
 expected=json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites']
 for p in sorted(Path('/Users/bwt/ac-worktrial/data/processed').glob('*.jsonl')):
  assert sha(p)==expected[p.stem]['sha256'];receipts.append({'file':p.name,'sha256':sha(p)})
  for l in p.open():
   for s in texts(json.loads(l)['state']):
    w=norm(s)
    if len(w)<8:continue
    h=rank(w);exact.add(h)
    if h in lengths:continue
    ss=shingles(w);lengths[h]=len(ss)
    for sh in ss:idx[sh].add(h)
 def hit(row):
  for s in texts([row['state'],row['question']]):
   w=norm(s)
   if len(w)<8:continue
   if rank(w) in exact:return True
   ss=shingles(w)
   if len(ss)<5:continue
   c=collections.Counter(e for sh in ss for e in idx.get(sh,()))
   if any(n>=5 and n/min(len(ss),lengths[e])>=.8 for e,n in c.items()):return True
  return False
 pins=json.loads((ROOT/'configs/training-source-pins.json').read_text());validator=Draft202012Validator(json.loads((ROOT/'configs/training-point-v2.schema.json').read_text()));added=[];sources={};reject=[]
 for source in ['ANLI','RACE','ReClor','MultiRC']:
  manifest=json.loads((ROOT/f'data/training/manifests/{source}.json').read_text());native=[];offset=0;raw=[]
  for f in manifest.get('files',manifest.get('raw_files',[])):
   path=ROOT/f.get('local_path',f['path']);assert sha(path)==f['sha256'];raw.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
   for i,r in enumerate(pq.read_table(path).to_pylist()):
    if source=='ANLI':
     split=Path(f['path']).name.split('-')[0];rid=f'ANLI:{split}:{offset}';offset+=1
     if r['label'] not in [0,1,2]:continue
     premise=r['premise'];n={'id':rid,'source':source,'source_revision':pins[source]['revision'],'source_split':split,'source_row_index':offset-1,'group_id':'ANLI:'+hashlib.sha256(premise.encode()).hexdigest(),'state':f"Premise: {premise}\nHypothesis: {r['hypothesis']}",'question':'How does the premise relate to the hypothesis?','options':{'0':'entailment','1':'neutral','2':'contradiction'},'target':str(r['label'])}
    else:
     file=str(path.relative_to(ROOT/'data/training/raw'/source));n=native_reading(source,file,i,r)
    if n['id'] not in oldids:native.append(n)
  selected=[];selected_ids=set()
  for n in sorted(native,key=lambda n:rank([SEED,source,n['id']])):
   if n['id'] in selected_ids:continue
   t=typed(n);a=adapt_training_point(t,validator);qid,q=next(iter(a['request']['questions'].items()));row={'id':n['id'],'suite':source,'state':a['request']['state'],'question':q,'label':t['gold'][qid],'group':a['group_id'],'source':t['native_source'],'evaluation_only':False}
   if hit(row):reject.append({'id':row['id'],'reason':'local benchmark overlap'});continue
   selected.append(row);selected_ids.add(row['id'])
   if len(selected)==500:break
  assert len(selected)==500;added.extend(selected);sources[source]={'added_ids':[r['id'] for r in selected],'eligible_remaining':len(native),'raw_files':raw}
 added.extend(r for r in hs if r['id'] not in oldids);assert len(added)==2500
 rows=old+added;assert len(rows)==5053 and len({r['id'] for r in rows})==5053
 additions=ROOT/'data/training/additions-2500-to-v2.jsonl';additions.write_text(''.join(canonical(r)+'\n' for r in added))
 target=ROOT/'data/training/train-5053-six-sources-quality-preference.jsonl';target.write_text(''.join(canonical(r)+'\n' for r in rows))
 # Full validation and proof that every old record survives unchanged.
 from benchmark_training_data import adapt_training_point as normalized
 v=Draft202012Validator(json.loads((ROOT/'configs/normalized-training.schema.json').read_text()))
 for r in rows:normalized(r,v)
 assert rows[:len(old)]==old
 report={'records':5053,'source_counts':dict(collections.Counter(r['suite'] for r in rows)),'types':dict(collections.Counter(r['question']['type'] for r in rows)),'seed':SEED,'helpsteer_seed':add_helpsteer2_pairwise.SEED,'schema_permutation_seed':20261016,'sampling':'Retain all 2553 existing records; take first 500 non-existing IDs per source ranked by SHA256([seed,source,id]) after screening. HelpSteer2 extends original SHA-ranked quality-only pair sample to 1000. No replacement.','original_sha256':sha(original),'output_sha256':sha(target),'sources':sources,'rejections':reject,'evaluation_files':receipts,'screen':'Exact normalized >=8 words; >=80% containment of 16-word shingles with >=5 shared shingles. Heuristic, not proof against paraphrases or shared ancestry.','additions_path':str(additions.relative_to(ROOT)),'additions_sha256':sha(additions),'added_records':2500,'existing_records_preserved':2553,'cache':'New rows require extraction; existing matching request hashes remain reusable.'}
 (ROOT/'outputs/training-expansion-5053-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 config=json.loads((ROOT/'configs/active-training-dataset.json').read_text());config.update({'dataset_version':'v2-expanded-1000','frozen_parent_version':'v2','release_manifest':'data/training/manifests/dataset-v2-expanded-1000.json','dataset_path':str(target.relative_to(ROOT)),'dataset_sha256':sha(target),'records':5053,'derivation_script':'scripts/expand_training_to_1000.py','expansion_audit':'outputs/training-expansion-5053-audit.json'});(ROOT/'configs/active-training-dataset.json').write_text(json.dumps(config,indent=2)+'\n')
 (ROOT/'data/training/manifests/dataset-v2-expanded-1000.json').write_text(json.dumps(dict(config,source_counts=report['source_counts'],seeds={'additional_sampling':SEED,'helpsteer':20261019,'schema_permutation':20261016}),indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['records','source_counts','types','added_records','existing_records_preserved','output_sha256']},indent=2))
if __name__=='__main__':main()
