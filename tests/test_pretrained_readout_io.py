"""No partial calibration/test attempt may be silently mixed into a result."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_pretrained_readout as module
from head_probe_io import write_json

class ReadoutIOTests(unittest.TestCase):
 def test_interrupted_attempt_retained_complete_attempt_published_once(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'readout';calls=[];request={'commit':'fixed','input_sha':'a'}
   def once(output,commit,checkpoint,run_name=None):
    output=Path(output);output.mkdir();calls.append(output)
    if len(calls)==1:
     (output/'calibration-logits.jsonl').write_text('partial first attempt\n');raise RuntimeError('simulated preemption')
    write_json(output/'completion.json',{'status':'complete','parameters_unchanged':True})
    (output/'calibrated-predictions.jsonl').write_text('complete second attempt\n')
    self.assertEqual(run_name,'readout');return {'status':'complete','run_id':output.name,'composite':.5}
   with patch.object(module,'_request_identity',return_value=request),patch.object(module,'_evaluate_once',side_effect=once):
    with self.assertRaisesRegex(RuntimeError,'simulated preemption'):module.evaluate(root,'fixed')
    result=module.evaluate(root,'fixed');self.assertEqual(result['run_id'],'readout');self.assertEqual((root/'calibrated-predictions.jsonl').read_text(),'complete second attempt\n')
    self.assertTrue((calls[0]/'calibration-logits.jsonl').exists());self.assertEqual(module.evaluate(root,'fixed'),result);self.assertEqual(len(calls),2)
 def test_attempt_cap_finishes_without_unbounded_replay(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'readout';request={'commit':'fixed'}
   with patch.object(module,'_request_identity',return_value=request),patch.object(module,'_evaluate_once',side_effect=RuntimeError('interrupted')) as mocked:
    for _ in range(3):
     with self.assertRaises(RuntimeError):module.evaluate(root,'fixed')
    self.assertEqual(module.evaluate(root,'fixed')['status'],'preemption_budget_exhausted');self.assertEqual(mocked.call_count,3)
if __name__=='__main__':unittest.main()
