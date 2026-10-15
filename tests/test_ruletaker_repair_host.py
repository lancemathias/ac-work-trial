import json,tempfile,unittest
from pathlib import Path
from run_ruletaker_repair_host import prerequisites,NAMES,digest
class RepairHandoff(unittest.TestCase):
 def blocked(self,phase,retired,containers,reason):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'outputs/autoresearch';folder.mkdir(parents=True);(folder/'math-source-grid-status.json').write_text(json.dumps({'phase':phase}))
   if retired:(folder/'math-worker-retired.json').write_text('{}')
   with self.assertRaisesRegex(RuntimeError,reason):prerequisites(root,containers)
 def test_live_grid_blocked(self):self.blocked('running',True,[],'Complete original')
 def test_live_keepalive_blocked(self):self.blocked('complete',False,[],'Retire original')
 def test_prior_gpu_blocked(self):self.blocked('complete',True,[{'app_name':'ac-worktrial-math-warm-research'}],'Prior GPU')
 def test_pretrained_gpu_blocked(self):self.blocked('complete',True,[{'app_name':'ac-worktrial-readout-order-warm'}],'Prior GPU')
 def exported(self,root):
  folder=root/'outputs/autoresearch';folder.mkdir(parents=True);(folder/'math-source-grid-status.json').write_text('{"phase":"complete"}');(folder/'math-worker-retired.json').write_text('{}')
  for name in NAMES:
   run='ab1-math-accuracy-seed20261014-'+name;artifact=root/'outputs/model-artifacts'/run;artifact.mkdir(parents=True);tensor=artifact/'best-head.safetensors';tensor.write_bytes(b'fixture selected tensor')
   meta=root/'outputs/training'/run;meta.mkdir(parents=True);(meta/'checksums.json').write_text(json.dumps({'best-head.safetensors':digest(tensor)}))
   test=root/'outputs/test'/(run+'-test');test.mkdir(parents=True)
   for filename in ['config.json','completion.json','metrics.json']:(test/filename).write_text('{"status":"complete"}')
   (test/'checksums.json').write_text(json.dumps({f.name:digest(f) for f in test.iterdir()}))
 def test_exported_complete_screen_passes(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.exported(root);self.assertTrue(prerequisites(root,[]))
 def test_changed_selected_tensor_blocks_handoff(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.exported(root);next((root/'outputs/model-artifacts').rglob('best-head.safetensors')).write_bytes(b'corrupt')
   with self.assertRaisesRegex(RuntimeError,'tensor not exported'):prerequisites(root,[])
if __name__=='__main__':unittest.main()

