import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from polarity_diagnostic import pairs,metrics
class PolarityTests(unittest.TestCase):
 def test_complement_and_evaluation_only_preserve_state(self):
  row={'id':'native','suite':'MultiRC','state':{'passage':'public evidence','candidate_answer':'answer'},'question':{'type':'noul','instructions':'Which event? Is the candidate answer correct?'},'label':True,'evaluation_only':False}
  original=copy.deepcopy(row);a,b=pairs([row]);self.assertEqual(row,original);self.assertEqual(a['state'],b['state']);self.assertTrue(a['label']);self.assertFalse(b['label']);self.assertTrue(a['evaluation_only'] and b['evaluation_only']);self.assertTrue(b['question']['instructions'].endswith('incorrect?'))
 def test_constant_false_is_not_instruction_sensitivity(self):
  rows=[{'diagnostic_parent_id':'a','polarity_variant':'original','label':True,'p_true':.1},{'diagnostic_parent_id':'a','polarity_variant':'negated','label':False,'p_true':.1}]
  result=metrics(rows);self.assertEqual(result['complement_consistency'],0);self.assertEqual(result['both_correct_fraction'],0)
  rows[0]['p_true']=.9;result=metrics(rows);self.assertEqual(result['complement_consistency'],1);self.assertEqual(result['both_correct_fraction'],1)
 def test_unsupported_template_rejected(self):
  with self.assertRaises(ValueError):pairs([{'suite':'MultiRC','question':{'type':'noul','instructions':'Ambiguous negation?'},'label':True}])
if __name__=='__main__':unittest.main()
