import math,unittest
from evaluate_cached_mixture import mixture,probabilities
class MixtureTests(unittest.TestCase):
 def pair(self):return ({'id':'x','type':'choice','option_ids':['a','b'],'calibrated_probabilities':[.8,.2],'target_index':0,'source':'allowed'},{'id':'x','type':'choice','option_ids':['b','a'],'logits':[0.,0.],'target_index':1,'source':'allowed'})
 def test_semantic_pool_is_equal_weight_and_normalized(self):
  a,b=self.pair();r=mixture([a],[b],1)[0];self.assertAlmostEqual(math.exp(r['logits'][0]),.65);self.assertAlmostEqual(math.exp(r['logits'][1]),.35);self.assertEqual(r['target_index'],0)
 def test_gold_source_and_pair_mismatches_rejected(self):
  a,b=self.pair();b['target_index']=0
  with self.assertRaises(ValueError):mixture([a],[b],1)
  a,b=self.pair();b['source']='different'
  with self.assertRaises(ValueError):mixture([a],[b],1)
  a,b=self.pair()
  with self.assertRaises(ValueError):mixture([a,a],[b],1)
 def test_test_mode_does_not_add_labels(self):
  a,b=self.pair();del a['target_index'];del b['target_index'];self.assertNotIn('target_index',mixture([a],[b],1)[0])
 def test_nonpositive_temperature_and_bad_probabilities_rejected(self):
  with self.assertRaises(ValueError):probabilities([1,2],0)
  a,b=self.pair();a['calibrated_probabilities']=[.8,.8]
  with self.assertRaises(ValueError):mixture([a],[b],1)
