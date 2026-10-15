import unittest
from thinking_staged_budget import gate
class StageBudget(unittest.TestCase):
 def capacity(self):return {'status':'complete','capacity_validated':True,'parameters_unchanged':True,'capacity_gate':{'measured_batch_seconds':[24.284276394,18.683712417,25.730705513]}}
 def test_dev_phase_fits_when_upfront_full_trial_did_not(self):
  v=gate(self.capacity(),'development',0,46474)
  self.assertEqual(v['status'],'passed');self.assertEqual(v['records'],1666);self.assertEqual(v['padded_records'],1672);self.assertAlmostEqual(v['estimated_trial_seconds'],25.730705513*418*2)
 def test_test_phase_rechecks_actual_remaining_time(self):
  self.assertEqual(gate(self.capacity(),'test',0,32000)['status'],'passed')
  self.assertEqual(gate(self.capacity(),'test',0,28000)['status'],'insufficient_reservation')
 def test_unverified_capacity_and_batch_change_rejected(self):
  v=self.capacity();v['capacity_validated']=False
  with self.assertRaises(ValueError):gate(v,'development',0,100000)
  with self.assertRaises(ValueError):gate(self.capacity(),'development',0,100000,batch_size=8)
if __name__=='__main__':unittest.main()
