"""Verify the delivery wrapper against the measured pooling and prompt contract."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from decision_model.inference import distribution,typed_answer,accepted_config
from decision_model._readout import messages
from decision_model._pool import pool

class DeliveryTests(unittest.TestCase):
 def test_semantic_pool_removes_letter_bias(self):
  left=[dict(id='x',type='choice',option_ids=['yes','no'],logits=[4.,0.])]
  right=[dict(id='x',type='choice',option_ids=['no','yes'],logits=[4.,0.])]
  result=pool(left,right)[0];p=distribution(result['logits'],accepted_config()['temperature'])
  self.assertAlmostEqual(p[0],.5);self.assertAlmostEqual(p[1],.5)
 def test_boolean_and_expected_score_semantics(self):
  self.assertEqual(typed_answer({'type':'noul'},['false','true'],[.3,.7]),{'type':'noul','noul':.7})
  answer=typed_answer({'type':'score','criteria':['low','medium','high']},['0','1','2'],[.1,.7,.2]);self.assertAlmostEqual(answer['score'],1.1)
 def test_prompt_equals_archived_implementation(self):
  from pretrained_readout import messages as archived
  q={'type':'choice','instructions':'Select','criteria':{'z':'later','a':'earlier'}}
  for order in ['native','reverse']:self.assertEqual(messages({'evidence':'only'},q,order),archived({'evidence':'only'},q,order))
 def test_saved_logits_reproduce_accepted_predictions(self):
  root=Path(__file__).resolve().parents[1];run=root/'outputs/model-artifacts/readout-native-reverse-pool-v1'
  if not (run/'calibrated-predictions.jsonl').exists():self.skipTest('Restore the archived trace bundle for integration verification')
  native=root/'outputs/model-artifacts/readout-fast-qwen38-27b-v1/test-logits.jsonl'
  reverse=run/'reverse-test-logits.jsonl'
  if not native.exists() or not reverse.exists():self.skipTest('Restore native and reverse logit traces')
  original=[json.loads(l) for l in native.read_text().splitlines()];reversed_order=[json.loads(l) for l in reverse.read_text().splitlines()]
  actual={r['id']:r for r in pool(original,reversed_order)}
  expected=[json.loads(l) for l in (run/'calibrated-predictions.jsonl').read_text().splitlines()]
  for row in expected:
   if row['status']!='ok':continue
   item=actual[row['id']];p=dict(zip(item['option_ids'],distribution(item['logits'],accepted_config()['temperature'])))
   for option,value in zip(row['options'],row['probabilities']):
    key=str(option).lower() if type(option) is bool else str(option)
    self.assertAlmostEqual(p[key],value,places=12)
