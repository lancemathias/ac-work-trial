import unittest,copy
from selective_thinking import winner,routed_ids,combine
class SelectiveTests(unittest.TestCase):
 def fixture(self):
  n=[dict(id='agree',type='noul',option_ids=['true','false'],logits=[2.,0.]),dict(id='disagree',type='noul',option_ids=['true','false'],logits=[2.,0.])]
  r=[dict(id='agree',type='noul',option_ids=['false','true'],logits=[0.,2.]),dict(id='disagree',type='noul',option_ids=['false','true'],logits=[3.,0.])]
  t=[dict(id='disagree',type='noul',option_ids=['false','true'],logits=[0.,4.])];return n,r,t
 def test_semantic_disagreement_and_display_independent_ties(self):
  n,r,t=self.fixture();self.assertEqual(routed_ids(n,r),['disagree']);self.assertEqual(winner(dict(option_ids=['z','a'],logits=[1.,1.])),'a');self.assertEqual(winner(dict(option_ids=['a','z'],logits=[1.,1.])),'a')
 def test_agreement_pool_and_semantic_thinking_alignment(self):
  from readout_order_pool import pool
  n,r,t=self.fixture();before=copy.deepcopy((n,r,t));out=combine(n,r,t);self.assertEqual(out[0]['logits'],pool(n,r)[0]['logits']);self.assertFalse(out[0]['thinking_routed']);self.assertTrue(out[1]['thinking_routed']);self.assertGreater(out[1]['logits'][0],out[1]['logits'][1]);self.assertEqual((n,r,t),before)
 def test_missing_duplicate_or_foreign_rows_rejected(self):
  n,r,t=self.fixture()
  for bad in [[],t+t,[dict(t[0],id='foreign')]]:
   with self.assertRaises(ValueError):combine(n,r,bad)
 def test_labels_do_not_affect_routing_but_mismatch_is_rejected(self):
  n,r,t=self.fixture();n[1]['target_index']=0;r[1]['target_index']=1;t[0]['target_index']=1
  self.assertEqual(routed_ids(n,r),['disagree']);combine(n,r,t);t[0]['target_index']=0
  with self.assertRaises(ValueError):combine(n,r,t)
if __name__=='__main__':unittest.main()
