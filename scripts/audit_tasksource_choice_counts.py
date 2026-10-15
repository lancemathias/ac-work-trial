"""Inventory only: scan source/kind/options columns; never add rows to training."""
import collections,concurrent.futures,hashlib,json,pathlib,urllib.request
import pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'outputs/tasksource-source-audit'
def main():
 rev=json.loads((P/'revision.json').read_text())['revision'];manifest=json.loads((P/'filtered-full/filtering-report.json').read_text());tree=json.loads((P/'filtered-full-pinned-tree.json').read_text());actual={r['path']:r for r in tree if r['path'].endswith('.parquet')};raw=P/'filtered-parquet';raw.mkdir(exist_ok=True)
 def get(s):
  p=raw/pathlib.Path(s['path']).name
  if not p.exists():urllib.request.urlretrieve(f'https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/resolve/{rev}/{s["path"]}',p)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==actual[s['path']]['lfs']['oid'];return p
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(get,manifest['shards']))
 counts=collections.defaultdict(collections.Counter)
 for p in paths:
  t=pq.read_table(p,columns=['source','kind','options'])
  for source,kind,opts in zip(t['source'].to_pylist(),t['kind'].to_pylist(),t['options'].to_pylist()):
   if kind=='choice':counts[source][len(opts)]+=1
 report={s:{'choice_questions':sum(c.values()),'option_counts':dict(sorted(c.items())),'five_to_eight':sum(n for k,n in c.items() if 5<=k<=8)} for s,c in sorted(counts.items())}
 (P/'filtered-choice-counts.json').write_text(json.dumps({'revision':rev,'sources':report,'historical_report_checksums_match_current':False,'file_verification':'Verified against LFS SHA256 at pinned revision; historical filtering-report shard checksums are stale.','scope':'All filtered-full Choice rows; source/kind/options columns only; not training admission or contamination clearance.'},indent=2)+'\n')
 for s,v in sorted(report.items(),key=lambda x:-x[1]['five_to_eight'])[:40]:print(s,json.dumps(v))
if __name__=='__main__':main()
