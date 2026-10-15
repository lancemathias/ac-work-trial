"""Sealed development-only correctness/incorrectness pairs and paired metrics."""
import collections,copy,json
CORRECT='Is the candidate answer correct?';INCORRECT='Is the candidate answer incorrect?'
def pairs(rows):
 out=[]
 for row in rows:
  if row['question']['type']!='noul':continue
  if row['suite']!='MultiRC' or not row['question']['instructions'].endswith(CORRECT) or type(row['label']) is not bool:raise ValueError('Unsupported Boolean diagnostic template')
  for variant in ['original','negated']:
   r=copy.deepcopy(row);r.update(id='polarity-'+variant+':'+row['id'],evaluation_only=True,diagnostic_parent_id=row['id'],polarity_variant=variant,role='development_diagnostic_only')
   if variant=='negated':r['question']['instructions']=row['question']['instructions'][:-len(CORRECT)]+INCORRECT;r['label']=not row['label']
   out.append(r)
 return out
def metrics(predictions):
 grouped=collections.defaultdict(dict)
 for p in predictions:
  if p['polarity_variant'] in grouped[p['diagnostic_parent_id']]:raise ValueError('Duplicate polarity observation')
  grouped[p['diagnostic_parent_id']][p['polarity_variant']]=p
 if not grouped:raise ValueError('Empty polarity probe')
 both=flip=original=negated=0;errors=[];false_counts=collections.Counter()
 for group in grouped.values():
  if set(group)!={'original','negated'}:raise ValueError('Incomplete polarity pair')
  a,b=group['original'],group['negated']
  if a['label'] is b['label']:raise ValueError('Pair labels must be complements')
  pa=a['p_true']>=.5;pb=b['p_true']>=.5;ca=pa==a['label'];cb=pb==b['label']
  original+=ca;negated+=cb;both+=ca and cb;flip+=pa!=pb;errors.append(abs(a['p_true']+b['p_true']-1));false_counts['original']+=not pa;false_counts['negated']+=not pb
 n=len(grouped)
 return {'pairs':n,'original_accuracy':original/n,'negated_accuracy':negated/n,'both_correct_fraction':both/n,'complement_consistency':flip/n,'mean_probability_complement_error':sum(errors)/n,'false_prediction_counts':dict(false_counts)}
if __name__=='__main__':
 import argparse
 from pathlib import Path
 p=argparse.ArgumentParser();p.add_argument('dev');p.add_argument('output');a=p.parse_args();rows=[json.loads(x) for x in Path(a.dev).open()];probe=pairs(rows);Path(a.output).write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n' for r in probe));print(json.dumps({'rows':len(probe),'pairs':len(probe)//2,'role':'development_diagnostic_only','evaluation_only':True}))
