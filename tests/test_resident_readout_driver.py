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
if __name__=='__main__':unittest.main()
