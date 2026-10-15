import unittest,copy
from fewshot_capacity import select
class CapacityTests(unittest.TestCase):
 def rows(self):return [dict(id=str(i),type='noul',target_index=i%2,source='allowed',encoding=dict(input_ids=list(range(10+i)),option_ids=['true','false'],code_token_ids=[1,2])) for i in range(20)]
 def test_quantiles_and_synthetic_cap_discard_labels_without_mutation(self):
  rows=self.rows();original=copy.deepcopy(rows);chosen=select(rows);self.assertEqual(len(chosen),12);self.assertEqual(len({r['id'] for r in chosen}),12);self.assertEqual([len(r['encoding']['input_ids']) for r in chosen[-4:]],[12288]*4);self.assertTrue(all(set(r)=={'id','type','encoding'} for r in chosen));self.assertEqual(rows,original);self.assertEqual(chosen[0]['id'],'0');self.assertEqual(chosen[7]['id'],'19')
 def test_invalid_cap_batch_small_or_overlength_input_rejected(self):
  for kw in [{'maximum':12287},{'batch_size':2}]:
   with self.assertRaises(ValueError):select(self.rows(),**kw)
  for rows in [self.rows()[:7],[dict(r,encoding=dict(r['encoding'],input_ids=list(range(12289)))) for r in self.rows()]]:
   with self.assertRaises(ValueError):select(rows)
if __name__=='__main__':unittest.main()
