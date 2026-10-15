"""Verified paired composite deltas; native-group bootstrap, fixed task weighting."""
import argparse,collections,json,platform,random,statistics
from pathlib import Path

def correct(row):return int(row['status']=='ok' and row.get('correct') is True)
def aggregate(strata):
 suites=collections.defaultdict(list)
 for (suite,subtask),value in strata.items():suites[suite].append(value)
 per_suite={suite:statistics.mean(values) for suite,values in suites.items()}
 return statistics.mean(per_suite.values()),per_suite

def compare(reference,candidate,repetitions=2000,seed=20261014):
 a={r['id']:r for r in reference};b={r['id']:r for r in candidate}
 if not a or len(a)!=len(reference) or len(b)!=len(candidate) or set(a)!=set(b):raise ValueError('Missing/duplicate/unpaired IDs')
 strata=collections.defaultdict(list);wins=losses=0
 for rid in sorted(a):
  x,y=a[rid],b[rid]
  for key in ['suite','subtask','group','label']:
   if type(x.get(key)) is not type(y.get(key)) or x.get(key)!=y.get(key):raise ValueError('Paired metadata mismatch: '+key)
  if x['status'] not in {'ok','error'} or y['status'] not in {'ok','error'}:raise ValueError('Unsupported result status')
  delta=correct(y)-correct(x);wins+=delta>0;losses+=delta<0
  strata[(x['suite'],x.get('subtask') if x['suite']=='BBH' else '')].append((x,y,delta))
 ref_point={key:statistics.mean(correct(x) for x,_,_ in values) for key,values in strata.items()};new_point={key:statistics.mean(correct(y) for _,y,_ in values) for key,values in strata.items()}
 ra,rs=aggregate(ref_point);ca,cs=aggregate(new_point)
 clusters={}
 for key,values in strata.items():
  groups=collections.defaultdict(list)
  for x,y,delta in values:groups[x.get('group') or x['id']].append(delta)
  clusters[key]=[(sum(v),len(v)) for _,v in sorted(groups.items())]
 if repetitions<1:raise ValueError('Positive bootstrap repetitions required')
 rng=random.Random(seed);boot=[]
 for _ in range(repetitions):
  draw={}
  for key,groups in clusters.items():
   sampled=[rng.choice(groups) for _ in groups];draw[key]=sum(x[0] for x in sampled)/sum(x[1] for x in sampled)
  boot.append(aggregate(draw)[0])
 boot.sort()
 def quantile(q):
  index=q*(len(boot)-1);low=int(index);high=min(low+1,len(boot)-1);return boot[low]+(index-low)*(boot[high]-boot[low])
 return {'reference_composite':ra,'candidate_composite':ca,'composite_delta':ca-ra,'paired_wins':wins,'paired_losses':losses,'paired_ties':len(a)-wins-losses,'records':len(a),'per_suite':{s:{'reference':rs[s],'candidate':cs[s],'delta':cs[s]-rs[s]} for s in sorted(rs)},'paired_cluster_bootstrap_95_interval':[quantile(.025),quantile(.975)],'bootstrap':{'repetitions':repetitions,'seed':seed,'python':platform.python_version(),'units':'Native groups resampled within suite and BBH-subtask strata; variable group sizes retained. Each suite gets equal weight, with BBH subtasks averaged first.','limitations':'Conditional paired sampling uncertainty on an observed, repeatedly used panel. Does not cover seed variability, multiple comparisons or selection bias.'}}

def verified(folder):
 from run_baselines import ROOT,load_panel,sha,score
 folder=Path(folder);checks=json.loads((folder/'checksums.json').read_text())
 for name in ['config.json','completion.json','calibrated-predictions.jsonl']:
  if name not in checks or sha(folder/name)!=checks[name]:raise ValueError('Artifact checksum mismatch: '+name)
 cfg=json.loads((folder/'config.json').read_text());receipt=json.loads((folder/'completion.json').read_text())
 if cfg['panel_sha256']!=sha(ROOT/'data/manifests/baseline-panel.json') or receipt['status']!='complete' or not receipt['parameters_unchanged']:raise ValueError('Incomplete or changed evaluation')
 originals={r['id']:r for r in load_panel()};rows=[json.loads(x) for x in (folder/'calibrated-predictions.jsonl').read_text().splitlines()]
 if len(rows)!=len(originals) or {r['id'] for r in rows}!=set(originals):raise ValueError('Incomplete fixed panel')
 for row in rows:
  original=originals[row['id']]
  for key in ['suite','subtask','group','label']:
   expected=original['source'].get('subtask') if key=='subtask' else original.get(key)
   if type(row.get(key)) is not type(expected) or row.get(key)!=expected:raise ValueError('Panel metadata mismatch: '+key)
  if row['status']=='ok':
   rebuilt=score(original,row['response'])
   if type(row['prediction']) is not type(rebuilt['prediction']) or row['prediction']!=rebuilt['prediction'] or row['correct'] is not rebuilt['correct']:raise ValueError('Stored answer score mismatch')
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('output');p.add_argument('--repetitions',type=int,default=2000);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Preserve existing comparison; use a fresh output path')
 result=compare(verified(a.reference),verified(a.candidate),a.repetitions);result['artifacts']={'reference':str(Path(a.reference).resolve()),'candidate':str(Path(a.candidate).resolve())};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['reference_composite','candidate_composite','composite_delta','paired_cluster_bootstrap_95_interval','paired_wins','paired_losses']},indent=2))
