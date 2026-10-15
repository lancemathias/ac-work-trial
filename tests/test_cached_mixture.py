import math,unittest
from evaluate_cached_mixture import mixture,probabilities
from apply_cached_mixture_test import parent_probabilities
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

class ParentProbabilityTests(unittest.TestCase):
 def test_rounded_float32_parent_matches_original_scorer_normalization(self):
  from run_baselines import score
  row={'question':{'type':'choice','criteria':{'False':'no','True':'yes'}},'label':'False'}
  body={'answers':{'q':{'type':'choice','probabilities':{'False':.5556701421737671,'True':.4443298280239105}}}}
  scored=score(row,body);actual=parent_probabilities({'response':body},['True','False'])
  self.assertEqual(actual,list(reversed(scored['probabilities'])))
  self.assertAlmostEqual(sum(actual),1.)
 def test_noul_maps_semantic_ids_and_invalid_parent_is_rejected(self):
  body={'answers':{'q':{'type':'noul','noul':.7}}}
  self.assertEqual(parent_probabilities({'response':body},['false','true']),[1-.7,.7])
  body={'answers':{'q':{'type':'choice','probabilities':{'a':.9,'b':.9}}}}
  with self.assertRaises(ValueError):parent_probabilities({'response':body},['a','b'])
