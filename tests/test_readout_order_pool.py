import copy,math,unittest
from readout_order_pool import pool
class OrderPoolTests(unittest.TestCase):
 def rows(self):return ({'id':'x','type':'noul','option_ids':['true','false'],'logits':[2.,0.],'target_index':0,'source':'allowed'}, {'id':'x','type':'noul','option_ids':['false','true'],'logits':[2.,0.],'target_index':1,'source':'allowed'})
 def test_position_bias_cancels_after_semantic_alignment(self):
  a,b=self.rows();before=copy.deepcopy([a,b]);r=pool([a],[b])[0]
  self.assertAlmostEqual(math.exp(r['logits'][0]),.5);self.assertAlmostEqual(math.exp(r['logits'][1]),.5);self.assertEqual(r['target_index'],0);self.assertEqual([a,b],before)
 def test_semantically_equal_predictions_survive_reversal(self):
  a,b=self.rows();b['logits']=[0.,2.];r=pool([a],[b])[0]
  self.assertAlmostEqual(math.exp(r['logits'][0]),1/(1+math.exp(-2)))
 def test_test_inputs_do_not_require_or_add_labels(self):
  a,b=self.rows();del a['target_index'];del b['target_index'];self.assertNotIn('target_index',pool([a],[b])[0])
 def test_extreme_logits_stay_finite(self):
  a,b=self.rows();a['logits']=[10000.,-10000.];b['logits']=[10000.,-10000.]
  self.assertTrue(all(math.isfinite(x) for x in pool([a],[b])[0]['logits']))
 def test_unpaired_duplicate_order_and_target_mismatches_rejected(self):
  a,b=self.rows()
  with self.assertRaises(ValueError):pool([a,a],[b])
  with self.assertRaises(ValueError):pool([a],[])
  b['target_index']=0
  with self.assertRaises(ValueError):pool([a],[b])
  a,b=self.rows();b['option_ids']=['true','false']
  with self.assertRaises(ValueError):pool([a],[b])
