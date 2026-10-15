import copy,unittest
from apply_selective_cached_test import require_development
class GateTests(unittest.TestCase):
 def fit(self):return dict(status='dev_gate_passed',selection_metric='accuracy',gate=dict(status='passed',pooled_dev_accuracy=.8430160692212608,selective_dev_accuracy=.8630160692212608,observed_gain=.02,required_gain=.01,dev_decisions=809),temperature=dict(temperature=1.2))
 def test_locked_accuracy_gate_and_positive_calibration_required(self):
  self.assertEqual(require_development(self.fit()),1.2)
  for key,value in [('status','failed'),('required_gain',.009),('dev_decisions',808),('pooled_dev_accuracy',.82),('observed_gain',.009),('selective_dev_accuracy',float('nan')),('selection_metric','nll')]:
   x=self.fit();x['gate'][key]=value
   with self.assertRaises(ValueError):require_development(x)
  for temperature in [0,-1,float('inf')]:
   x=self.fit();x['temperature']['temperature']=temperature
   with self.assertRaises(ValueError):require_development(x)
 def test_status_or_inconsistent_accuracy_cannot_bypass_gate(self):
  x=self.fit();x['status']='dev_gate_failed'
  with self.assertRaises(ValueError):require_development(x)
  x=self.fit();x['gate']['observed_gain']=.03
  with self.assertRaises(ValueError):require_development(x)
if __name__=='__main__':unittest.main()
