import hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_predictions import verified
class VerificationTests(unittest.TestCase):
 def test_rehashed_false_score_and_changed_bytes_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'data/manifests').mkdir(parents=True);panel=root/'data/manifests/baseline-panel.json';panel.write_text('{}')
   sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
   original={'id':'x','suite':'X','source':{'subtask':None},'group':'g','label':True}
   stub=types.SimpleNamespace(ROOT=root,load_panel=lambda:[original],sha=sha,score=lambda r,response:{'prediction':True,'correct':True})
   row={'id':'x','suite':'X','subtask':None,'group':'g','label':True,'status':'ok','prediction':True,'correct':True,'response':{}}
   (root/'config.json').write_text(json.dumps({'panel_sha256':sha(panel)}));(root/'completion.json').write_text(json.dumps({'status':'complete','parameters_unchanged':True}))
   pred=root/'calibrated-predictions.jsonl'
   def save():
    pred.write_text(json.dumps(row)+'\n');(root/'checksums.json').write_text(json.dumps({n:sha(root/n) for n in ['config.json','completion.json','calibrated-predictions.jsonl']}))
   with patch.dict(sys.modules,{'run_baselines':stub}):
    save();self.assertEqual(len(verified(root)),1)
    row['correct']=False;save()
    with self.assertRaisesRegex(ValueError,'Stored answer score'):verified(root)
    row['correct']=True;save();pred.write_text(pred.read_text()+'\n')
    with self.assertRaisesRegex(ValueError,'checksum'):verified(root)
if __name__=='__main__':unittest.main()
