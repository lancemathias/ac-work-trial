import copy,unittest
from assess_selective_thinking import assess
import test_selective_thinking
class DevelopmentTests(unittest.TestCase):
 def inputs(self):
  n,r,t=test_selective_thinking.SelectiveTests().fixture()
  for x in n:x['target_index']=0
  for x in r:x['target_index']=1
  t[0]['target_index']=1
  cal=copy.deepcopy((n,r,t))
  for rows in cal:
   for x in rows:x['id']='cal-'+x['id']
  return (*cal,n,r,t)
 def test_accuracy_gain_advances_without_test_input(self):
  value,cal,dev=assess(*self.inputs());self.assertEqual(value['status'],'dev_gate_passed');self.assertEqual(value['gate']['pooled_dev_accuracy'],.5);self.assertEqual(value['gate']['selective_dev_accuracy'],1.);self.assertFalse(value['test_read']);self.assertEqual(value['routing'],{'calibration':1,'dev':1})
 def test_no_gain_rejected_independent_of_confidence_magnitude(self):
  x=list(self.inputs());x[-1][0]['logits']=[100.,0.];value,_,_=assess(*x);self.assertEqual(value['status'],'dev_gate_failed');self.assertEqual(value['gate']['observed_gain'],0.)
 def test_calibration_dev_overlap_rejected(self):
  x=list(self.inputs());x[0][0]['id']='agree';x[1][0]['id']='agree'
  with self.assertRaises(ValueError):assess(*x)
if __name__=='__main__':unittest.main()
