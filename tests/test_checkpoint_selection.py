import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from checkpoint_selection import best_epoch,selection_value
class SelectionTests(unittest.TestCase):
 def test_accuracy_wins_when_nll_disagrees_and_ties_keep_earlier(self):
  rows=[{'event':'dev_epoch','epoch':1,'accuracy':.8,'nll':.6},{'event':'dev_epoch','epoch':2,'accuracy':.7,'nll':.4},{'event':'dev_epoch','epoch':3,'accuracy':.8,'nll':.3}]
  self.assertEqual(best_epoch(rows)['epoch'],1);self.assertEqual(best_epoch(rows,'nll')['epoch'],3)
  self.assertLess(selection_value(rows[0],'accuracy'),selection_value(rows[1],'accuracy'))
 def test_no_test_or_calibration_rows_can_select_checkpoint(self):
  with self.assertRaisesRegex(ValueError,'No development'):best_epoch([{'event':'final_test','accuracy':1.}])
  with self.assertRaisesRegex(ValueError,'Unsupported'):selection_value({},'test_accuracy')
if __name__=='__main__':unittest.main()
