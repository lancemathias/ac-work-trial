import json,tempfile,unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock,patch
import run_resident_readout_trials as driver

class HandoffChecks(unittest.TestCase):
 def check_block(self,phase,retired,containers,expected):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'outputs/autoresearch';folder.mkdir(parents=True)
   (folder/'math-source-grid-status.json').write_text(json.dumps({'phase':phase}))
   if retired:(folder/'math-worker-retired.json').write_text('{}')
   modal=ModuleType('modal');modal.Cls=Mock();config=ModuleType('modal.config');config._set_profile=Mock()
   with patch.dict('sys.modules',{'modal':modal,'modal.config':config}),patch.object(driver,'ROOT',root),patch.object(driver.subprocess,'check_output',return_value=json.dumps(containers)):
    with self.assertRaisesRegex(RuntimeError,expected):driver.main()
   modal.Cls.from_name.assert_not_called()
 def test_incomplete_grid_never_requests_worker(self):self.check_block('running',True,[],'Finish original source grid')
 def test_active_keepalive_never_requests_worker(self):self.check_block('complete',False,[],'Retire original host keepalive')
 def test_existing_gpu_never_requests_second_worker(self):self.check_block('complete',True,[{'app_name':'ac-worktrial-math-warm-research'}],'Old GPU worker remains')
 def test_repair_gpu_never_requests_second_worker(self):self.check_block('complete',True,[{'app_name':'ac-worktrial-ruletaker-repair-warm'}],'Old GPU worker remains')
 def test_repair_keepalive_must_retire_before_next_worker(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp)/'outputs/autoresearch';folder.mkdir(parents=True);(folder/'ruletaker-repair-call.json').write_text('{"request":{"run_id":"repair"}}')
   with self.assertRaisesRegex(RuntimeError,'Retire repair host keepalive'):driver.check_repair_handoff(folder)
 def test_repair_outputs_must_be_exported(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp)/'outputs/autoresearch';folder.mkdir(parents=True);(folder/'ruletaker-repair-call.json').write_text('{"request":{"run_id":"repair"}}');(folder/'repair-worker-retired.json').write_text('{}');(folder/'ruletaker-repair-result.json').write_text('{"status":"complete"}')
   with self.assertRaisesRegex(RuntimeError,'Export repair training'):driver.check_repair_handoff(folder)
if __name__=='__main__':unittest.main()

