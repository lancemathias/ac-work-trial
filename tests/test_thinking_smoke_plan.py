import unittest
from thinking_smoke_plan import select,reservation_gate

class SmokePlan(unittest.TestCase):
 def test_label_blind_quantiles_and_extremes(self):
  rows=[dict(id=str(i),type='Boolean',encoding={'input_ids':[1]*(i+1)},target_index=i%2,source='private') for i in range(20)]
  chosen=select(rows)
  self.assertEqual(len(chosen),8)
  self.assertEqual(chosen[0]['id'],'0');self.assertEqual(chosen[-1]['id'],'19')
  self.assertTrue(all(set(x)=={'id','type','encoding'} for x in chosen))
  self.assertEqual(chosen,select([{**x,'target_index':999} for x in rows]))
 def test_small_population_no_duplicate(self):
  self.assertEqual(len(select([dict(id='a',type='Choice',encoding={'input_ids':[1]})])),1)
 def test_budget_includes_idle_margin_and_safety(self):
  result=reservation_gate([2.,3.],100,10000,records=8)
  self.assertEqual(result['required_remaining_seconds'],1812)
  self.assertEqual(result['status'],'passed')
  self.assertEqual(reservation_gate([3.],100,1912,records=8)['status'],'insufficient_reservation')
 def test_invalid_measurement(self):
  for times in ([],[float('nan')],[0],[-1]):
   with self.assertRaises(ValueError):reservation_gate(times,0,1000)

if __name__=='__main__':unittest.main()
