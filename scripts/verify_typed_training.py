"""Validate the outer training-point contract and every schema/target pair."""
import collections,hashlib,importlib.metadata,json,pathlib,sys,tempfile
from generate_typed_training import generate,canonical,validate
from jsonschema import Draft202012Validator
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 config=json.loads((ROOT/'configs/typed-generator-config.json').read_text())
 for path,key in [('data/training/sample-5000.jsonl','input_sha256'),('data/training/typed-5000.jsonl','expected_typed_sha256'),('configs/training-point.schema.json','expected_schema_sha256'),('scripts/generate_typed_training.py','expected_generator_sha256')]:
  assert sha(ROOT/path)==config[key],f'Frozen artifact drift: {path}'
 schema_path=ROOT/'configs/training-point.schema.json';schema=json.loads(schema_path.read_text());Draft202012Validator.check_schema(schema);outer=Draft202012Validator(schema)
 p=ROOT/'data/training/typed-5000.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==5000 and len({r['id'] for r in rows})==5000
 source_counts=collections.Counter();types=collections.Counter()
 for r in rows:
  outer.validate(r);validate(r);s=r['model_payload']['schema'];Draft202012Validator.check_schema(s);Draft202012Validator(s).validate(r['gold'])
  assert set(s['properties'])==set(s['required'])==set(r['gold'])==set(r['field_types'])
  assert set(json.loads(r['model_payload']['input']))=={'state','question'}
  for k,t in r['field_types'].items():
   assert (t=='noul')==(s['properties'][k]['type']=='boolean')
   if t=='score':assert s['properties'][k]['enum']==['0','1','2','3','4']
  source_counts[r['source']]+=1;types.update(r['field_types'].values())
 assert set(source_counts.values())=={250}
 with tempfile.TemporaryDirectory() as d:
  generated=pathlib.Path(d)/'typed.jsonl';m=pathlib.Path(d)/'manifest.json';generate(ROOT/'data/training/sample-5000.jsonl',generated,m)
  assert generated.read_bytes()==p.read_bytes(),'Typed byte drift on regeneration'
 receipt={'record_count':len(rows),'field_count':sum(types.values()),'source_counts':dict(sorted(source_counts.items())),'field_type_counts':dict(types),'typed_sha256':sha(p),'training_point_schema_sha256':sha(schema_path),'generator_sha256':sha(ROOT/'scripts/generate_typed_training.py'),'python_version':sys.version.split()[0],'dependencies':{p:importlib.metadata.version(p) for p in ['pyarrow','jsonschema','attrs','jsonschema-specifications','referencing','rpds-py']},'byte_identical_regeneration':True,'all_outer_and_gold_schemas_valid':True,'confidence_fields':False,'model_calls':0,'nimble_calls':0,'schema_seed':20261016}
 out=ROOT/'data/training/manifests/typed-validation.json';out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
