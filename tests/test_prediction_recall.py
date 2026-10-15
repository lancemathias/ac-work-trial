import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_prediction_dominance import summarize
class RecallTests(unittest.TestCase):
 def test_misses_lower_recall_without_becoming_predictions(self):
  rows=[{'suite':'X','label':True,'prediction':True,'correct':True,'status':'ok'}, {'suite':'X','label':True,'status':'error'}, {'suite':'X','label':False,'prediction':True,'correct':False,'status':'ok'}]
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'rows.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows));r=summarize(p)['slices']['X']
  self.assertEqual(r['recall_counting_misses_wrong'],{'true':.5,'false':0})
  self.assertEqual(r['prediction_counts'],{'true':2})
  self.assertEqual(r['confusion_counts']['true'],{'true':1,'<missing>':1})
if __name__=='__main__':unittest.main()
