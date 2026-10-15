import importlib.util,json,tempfile,unittest,decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('typed',ROOT/'scripts/generate_typed_training.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class TypedTraining(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(x) for x in (ROOT/'data/training/sample-5000.jsonl').read_text().splitlines()]
 def test_byte_identical_regeneration(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);mod.generate(ROOT/'data/training/sample-5000.jsonl',p/'a',p/'am');mod.generate(ROOT/'data/training/sample-5000.jsonl',p/'b',p/'bm');self.assertEqual((p/'a').read_bytes(),(p/'b').read_bytes());self.assertEqual((p/'a').read_bytes(),(ROOT/'data/training/typed-5000.jsonl').read_bytes())
 def test_choice_permutation_preserves_semantics(self):
  for r in self.rows:
   typed=mod.convert(r)
   for trace in typed['transformation_trace']:
    if trace['operation']!='choice_permutation':continue
    target=r['target'];options=r.get('input',r).get('options');source=r['source']
    if source=='GSM8K':key='0'
    elif source=='HelpSteer2':key=target['choice']
    elif isinstance(options,list):key=str(target)
    elif str(target) in options:key=str(target)
    else:key=next(k for k,v in options.items() if v==target)
    self.assertEqual(trace['original_to_typed'][key],typed['gold'][trace['field']])
 def test_critical_recasts_and_payload_separation(self):
  for r in self.rows:
   t=mod.convert(r);mod.validate(t);payload=t['model_payload'];self.assertEqual(set(payload),{'input','schema'});parsed=json.loads(payload['input']);self.assertEqual(set(parsed),{'state','question'})
   if r['source']=='CivilComments':self.assertEqual(t['gold']['decision'],r['target']>=0.5);self.assertEqual(t['target_metadata']['annotation_fractions']['toxicity'],r['target'])
   if r['source']=='RuleTaker':self.assertEqual(t['gold']['decision'],r['raw_native']['label']=='entailment')
   if r['source']=='HelpSteer2' and r['source_config']=='ratings':self.assertEqual(t['gold'],{k:str(v) for k,v in r['target'].items()});self.assertEqual(set(t['field_types'].values()),{'score'})
   if r['source']=='GSM8K':self.assertNotIn(r['raw_native']['answer'],payload['input'])
 def test_unknown_entailment_label_fails_closed(self):
  r=dict(next(r for r in self.rows if r['source']=='RuleTaker'));r['target']='unknown'
  with self.assertRaises(AssertionError):mod.convert(r)
 def test_decimal_context_and_score_anchors(self):
  r=next(r for r in self.rows if r['source']=='GSM8K');baseline=mod.canonical(mod.convert(r))
  with decimal.localcontext() as ctx:
   ctx.prec=2;self.assertEqual(baseline,mod.canonical(mod.convert(r)))
  r=next(r for r in self.rows if r['source']=='HelpSteer2' and r['source_config']=='ratings')
  for field,p in mod.convert(r)['model_payload']['schema']['properties'].items():
   for level in range(5):self.assertIn(str(level)+':',p['description'])
   if field in ['complexity','verbosity']:self.assertIn('not quality',p['description'])
 def test_hash_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaises(ValueError):mod.generate(ROOT/'data/training/sample-5000.jsonl',p/'out',p/'manifest',expected='wrong')
if __name__=='__main__':unittest.main()
