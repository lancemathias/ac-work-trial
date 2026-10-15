"""Materialize benchmark-shaped training records; preserve every target and option key."""
import collections, hashlib, json
from pathlib import Path
from jsonschema import Draft202012Validator
from training_data import adapt_training_point, canonical
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 config_path=ROOT/'configs/active-training-dataset.json';config=json.loads(config_path.read_text())
 source=ROOT/config.get('typed_source_path',config['dataset_path'])
 validator=Draft202012Validator(json.loads((ROOT/'configs/training-point-v2.schema.json').read_text()))
 rows=[];requests=[]
 for line in source.open():
  point=json.loads(line);adapted=adapt_training_point(point,validator)
  assert len(adapted['request']['questions'])==1
  qid,question=next(iter(adapted['request']['questions'].items()))
  assert isinstance(question['instructions'],str)
  assert adapted['targets'][qid]==('true' if point['gold'][qid] is True else 'false' if point['gold'][qid] is False else point['gold'][qid])
  if question['type']=='choice':
   assert list(question['criteria'])==point['model_payload']['schema']['properties'][qid]['enum']
   assert all(isinstance(v,str) and v for v in question['criteria'].values())
  rows.append({'id':point['id'],'suite':point['source'],'state':adapted['request']['state'],
               'question':question,'label':point['gold'][qid],'group':point['family_id'],
               'source':point['native_source'],'evaluation_only':False})
  requests.append({'id':point['id'],'request':{'model':'jev-latest','state':adapted['request']['state'],'questions':{'q':question}},'label':point['gold'][qid]})
 target=ROOT/'data/training/train-4303-benchmark-format.jsonl';target.write_text(''.join(canonical(r)+'\n' for r in rows))
 preview=ROOT/'outputs/training-benchmark-format-requests.jsonl';preview.write_text(''.join(canonical(r)+'\n' for r in requests))
 report={'records':len(rows),'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'output':str(target.relative_to(ROOT)),'output_sha256':sha(target),'adapter_sha256':sha(ROOT/'scripts/training_data.py'),'gold_labels_changed':0,'option_keys_changed':0,'schema':'Same record shape as normalized benchmarks; same baseline wrapper questions.q; instructions string; choice criteria maps option key directly to description.','question_types':dict(collections.Counter(r['question']['type'] for r in rows)),'randomness':'None. Existing option permutations retained.','cache_compatibility':'New model input tokens and option spans. Original hidden caches are incompatible with these requests; frozen cache adapter remains untouched.','verification':'All 4303 source records pass outer schema and gold schema validation; all option-key orders and gold values retained; all Choice descriptions nonempty strings.'}
 (ROOT/'outputs/training-benchmark-format-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 config.update({'typed_source_path':str(source.relative_to(ROOT)),'dataset_path':str(target.relative_to(ROOT)),'dataset_sha256':sha(target),'format':'normalized-benchmark-v1','request_wrapper':'run_baselines.payload: questions.q','adapter':'scripts/training_data.py','format_audit':'outputs/training-benchmark-format-audit.json'})
 config_path.write_text(json.dumps(config,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
