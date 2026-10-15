import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import finish_current_run_and_release as m
class ReleaseGuardTests(unittest.TestCase):
 def fixture(self,root):
  folder=root/'outputs/autoresearch';folder.mkdir(parents=True);(folder/'research-stop-after-current.json').write_text(json.dumps(dict(last_allowed_run=m.RUN,function_call_id=m.CALL,modal_app=m.APP_ID)));(folder/'staged-readout-launch.json').write_text(json.dumps(dict(modal_app=m.APP_ID)));(folder/'staged-readout-worker-retired.json').write_text('{}')
 def test_observation_failure_does_not_stop_gpu_or_export(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root);call=MagicMock();call.get.side_effect=ConnectionError('synthetic observation failure')
   with patch.object(m,'ROOT',root),patch('modal.config._set_profile'),patch('modal.FunctionCall.from_id',return_value=call),patch('export_training.export') as export,patch.object(m.subprocess,'run') as stop,patch.object(m.subprocess,'check_output') as inspect:
    with self.assertRaises(ConnectionError):m.main()
    export.assert_not_called();stop.assert_not_called();inspect.assert_not_called()
   self.assertEqual(json.loads((root/'outputs/autoresearch/gpu-release-status.json').read_text())['phase'],'observation_error_needs_terminal_verification')
 def test_pending_then_terminal_exports_before_exact_app_stop(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root);call=MagicMock();call.get.side_effect=[TimeoutError(),{'status':'rejected_on_dev'}];events=[]
   def exported(run,destination):
    events.append('export');destination.mkdir(parents=True);raw=b'preserved';(destination/'data.txt').write_bytes(raw);(destination/'checksums.json').write_text(json.dumps({'data.txt':hashlib.sha256(raw).hexdigest()}))
   def stopped(args,**kwargs):
    self.assertEqual(events,['export']);self.assertEqual(args[1:4],['app','stop',m.APP_ID]);events.append('stop');return SimpleNamespace(returncode=0)
   with patch.object(m,'ROOT',root),patch('modal.config._set_profile'),patch('modal.FunctionCall.from_id',return_value=call),patch('export_training.export',side_effect=exported),patch.object(m.subprocess,'run',side_effect=stopped),patch.object(m.subprocess,'check_output',side_effect=['[]',json.dumps([{'app_id':m.APP_ID,'state':'stopped'}])]):m.main()
   receipt=json.loads((root/'outputs/autoresearch/gpu-release-receipt.json').read_text());self.assertTrue(receipt['GPU_released']);self.assertTrue(receipt['persistent_volumes_retained']);self.assertEqual(events,['export','stop']);self.assertEqual(call.get.call_count,2)
if __name__=='__main__':unittest.main()
