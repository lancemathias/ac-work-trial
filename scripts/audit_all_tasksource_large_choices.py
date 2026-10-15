"""Exhaustive pinned source/option inventory; HTTP range reads avoid image payloads."""
import collections,concurrent.futures,json,pathlib,time
import fsspec,pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'outputs/tasksource-source-audit'
def main():
 rev=json.loads((P/'revision.json').read_text())['revision'];tasks=[]
 for config,folder in [('full','full'),('default','data'),('vision','vision')]:
  for entry in json.loads((P/(folder+'-pinned-tree.json')).read_text()):
   if entry['path'].endswith('.parquet'):tasks.append((config,entry))
 checkpoints=P/'column-audit-parts';checkpoints.mkdir(exist_ok=True)
 def read(task):
  config,entry=task;name=config+'-'+entry['path'].split('/')[-1]+'.json';dest=checkpoints/name
  if dest.exists():return json.loads(dest.read_text())
  url=f'https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/resolve/{rev}/{entry["path"]}'
  for attempt in range(3):
   try:
    with fsspec.open(url,'rb',block_size=65536) as stream:
     table=pq.read_table(stream,columns=['id','source','kind','options','question'])
    break
   except Exception:
    if attempt==2:raise
    time.sleep(2)
  eligible=[];sourcecounts=collections.Counter()
  for batch in table.to_batches(max_chunksize=16384):
   for row in batch.to_pylist():
    sourcecounts[row['source']]+=1
    if row['kind']=='choice' and len(row['options'])>=5:
     eligible.append({'id':row['id'],'source':row['source'],'n':len(row['options']),'question':row['question']})
  value={'config':config,'file':entry['path'],'repo_lfs_sha256':entry.get('lfs',{}).get('oid'),'rows':table.num_rows,'split':entry['path'].split('/')[-1].split('-')[0],'source_physical_counts':dict(sourcecounts),'eligible':eligible}
  tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(dest);return value
 sources=collections.defaultdict(lambda:collections.Counter());seen={};duplicates=collections.Counter();questions=collections.defaultdict(list);metadata=[];n=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
  futures=[pool.submit(read,t) for t in tasks]
  for future in concurrent.futures.as_completed(futures):
   part=future.result();n+=1;config=part['config'];split=part['split'];metadata.append({k:v for k,v in part.items() if k!='eligible'})
   for r in part['eligible']:
    key=(config,split,r['id']);sig=(r['source'],r['n'],r['question'])
    if key in seen:
     if seen[key]!=sig:raise ValueError(('Conflicting selected metadata for duplicate ID',key))
     duplicates[config]+=1;continue
    seen[key]=sig;sources[(config,r['source'],split)][r['n']]+=1
    if r['question'] not in questions[(config,r['source'])] and len(questions[(config,r['source'])])<3:questions[(config,r['source'])].append(r['question'])
   if n%20==0:print(json.dumps({'files_completed':n,'total_files':len(tasks),'unique_eligible_ids':len(seen)}),flush=True)
 result={'revision':rev,'scope':'Every Parquet matched by current full/default/vision config globs at pinned revision. Only Choice rows >=5 criteria. Counts deduplicate ID within config/split; include physical-file and duplicate evidence. No train admission/contamination clearance.','sources':[{'config':config,'source':source,'split':split,'examples_ge5':sum(c.values()),'option_counts_ge5':dict(sorted(c.items())),'question_examples':questions[(config,source)]} for (config,source,split),c in sorted(sources.items())],'duplicate_eligible_ids':dict(duplicates),'files':metadata,'verification':'Range reads pinned to immutable repository revision; LFS hashes recorded from tree; full shard bytes not downloaded or rehashed.'}
 (P/'all-large-choice-counts.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print('COMPLETE',len(result['sources']))
if __name__=='__main__':main()
