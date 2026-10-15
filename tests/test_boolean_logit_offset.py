import unittest,copy
from boolean_logit_offset import shifted,fit,dev_gate
class OffsetTests(unittest.TestCase):
 def test_semantic_true_only_and_no_input_mutation(self):
  row={'type':'noul','option_ids':['false','true'],'logits':[2.,1.],'target_index':1};before=copy.deepcopy(row)
  self.assertEqual(shifted(row,2)['logits'],[2.,3.]);self.assertEqual(row,before)
  row['type']='choice';self.assertEqual(shifted(row,2),row)
 def test_fit_rejects_dev_test_and_selects_smallest_tied_offset(self):
  row={'type':'noul','option_ids':['true','false'],'logits':[0.,1.],'target_index':0}
  self.assertEqual(fit([row],'calibration')['bias'],1.)
  for role in ['train','dev','test']:
   with self.assertRaises(ValueError):fit([row],role)
 def test_zero_bias_wins_when_already_correct(self):
  row={'type':'noul','option_ids':['true','false'],'logits':[2.,1.],'target_index':0}
  self.assertEqual(fit([row],'calibration')['bias'],0)
 def test_dev_gate_enforces_both_thresholds(self):
  row={'type':'noul','option_ids':['true','false'],'logits':[0.,1.],'target_index':0}
  self.assertEqual(dev_gate([row],0)['status'],'failed');self.assertEqual(dev_gate([row],1)['status'],'passed')
