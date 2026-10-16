"""Check whether the old synthetic numeric options reveal the training label."""
import argparse,collections,decimal,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(root):
 root=Path(root);path=root/'data/training/typed-5000.jsonl';manifest=json.loads((root/'data/training/manifests/typed-5000.json').read_text())
 if sha(path)!=manifest['output_sha256']:raise ValueError('Typed source changed')
 rows=[r for r in map(json.loads,path.open()) if r['source']=='GSM8K'];positions=collections.Counter();correct=0
 for row in rows:
  if len(row['gold'])!=1:raise ValueError('Expected one hard answer')
  field,target=next(iter(row['gold'].items()))
  trace=next(t for t in row['transformation_trace'] if t.get('operation')=='choice_permutation' and t['field']==field)
  options=trace['option_descriptions'];ordered=sorted(options,key=lambda k:decimal.Decimal(options[k]))
  if len(ordered)!=5:raise ValueError('Expected5 numeric alternatives')
  positions[str(ordered.index(target))]+=1;correct+=ordered[2]==target
 return {'status':'audited_not_used','records':len(rows),'options_only_median_rule_correct':correct,'options_only_median_rule_accuracy':correct/len(rows),'gold_rank_among_numeric_options':dict(positions),'typed_source_sha256':sha(path),'generator_sha256':sha(root/'scripts/legacy_generate_typed_training.py'),'mechanism':'Legacy generator creates answer plus offsets0,-2,-1,+1,+2, then permutes labels. Correct answer is always the numeric median; no question or reasoning needed.','decision':'Do not blindly restore this synthetic supervision. Require explicit data-quality recast/review before a lower-priority GSM8K experiment. No model fitting or benchmark predictions.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Preserve prior audit')
 result=audit(a.root);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
