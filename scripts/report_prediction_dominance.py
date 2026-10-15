"""Record prediction/gold distributions; dominance is a diagnostic, not proof."""
import argparse,collections,json,math
from pathlib import Path
def summarize(path):
 rows=[json.loads(line) for line in Path(path).open()];groups=collections.defaultdict(list)
 for row in rows:
  name=row['suite']+('/'+row['subtask'] if row['suite']=='BBH' and row.get('subtask') else '')
  groups[name].append(row)
 result={}
 for name,items in sorted(groups.items()):
  valid=[r for r in items if r['status']=='ok']
  counts=lambda key:dict(collections.Counter(json.dumps(r[key],sort_keys=True) for r in valid))
  predictions=counts('prediction');gold=counts('label');n=len(valid)
  dominance=max(predictions.values(),default=0)/n if n else None
  result[name]={'expected':len(items),'valid':n,'prediction_counts':predictions,'gold_counts_valid':gold,'dominant_prediction_fraction':dominance,'dominance_flag':dominance is not None and dominance>=.95,'prediction_entropy_nats':-sum((v/n)*math.log(v/n) for v in predictions.values()) if n else None,'correct':sum(bool(r['correct']) for r in valid),'misses':len(items)-n}
 return {'slices':result,'interpretation':'A >=95% single-label prediction rate is a flag for inspection, not an automatic collapse claim. Compare gold distribution, task labels, failures and paired baseline.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('predictions');p.add_argument('output');args=p.parse_args();Path(args.output).write_text(json.dumps(summarize(args.predictions),indent=2)+'\n')
