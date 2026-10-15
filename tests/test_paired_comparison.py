import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_predictions import compare

def row(rid,suite,subtask='',group=None,correct=False):return {'id':rid,'suite':suite,'subtask':subtask,'group':group or rid,'label':True,'status':'ok','correct':correct}
class PairedComparisonTests(unittest.TestCase):
 def test_bbh_subtasks_weighted_before_suite_macro(self):
  a=[row('a','Other'),row('b','BBH','small')]+[row(str(i),'BBH','large') for i in range(9)];b=copy.deepcopy(a);b[1]['correct']=True;r=compare(a,b,repetitions=30)
  self.assertEqual(r['candidate_composite'],.25);self.assertEqual(r['composite_delta'],.25);self.assertEqual(r['paired_wins'],1)
 def test_clustered_pairs_and_identical_predictions(self):
  a=[row('a','Suite',group='same',correct=False),row('b','Suite',group='same',correct=True)];b=copy.deepcopy(a);b[0]['correct']=True;b[1]['correct']=False
  self.assertEqual(compare(a,b,40)['paired_cluster_bootstrap_95_interval'],[0,0]);self.assertEqual(compare(a,a,40)['composite_delta'],0)
 def test_missing_is_wrong_and_labels_must_match(self):
  a=[row('a','Suite',correct=True)];b=copy.deepcopy(a);b[0]['status']='error';r=compare(a,b,10);self.assertEqual(r['composite_delta'],-1)
  b[0]['label']=False
  with self.assertRaisesRegex(ValueError,'metadata'):compare(a,b,10)
if __name__=='__main__':unittest.main()
