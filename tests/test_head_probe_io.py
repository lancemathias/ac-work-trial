import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head_probe_io import prepare_run,seal,write_json

class PreemptionTests(unittest.TestCase):
 def test_replay_after_empty_directory_and_partial_bootstrap(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'request.tmp').write_text('{partial');request={'commit':'a','seed':7}
   self.assertIsNone(prepare_run(root,request));self.assertEqual(json.loads((root/'request.json').read_text()),request)
   self.assertEqual(len(list(root.glob('preempted-bootstrap-*'))),1)
   (root/'attempts').mkdir();(root/'attempts'/'interrupted').mkdir();self.assertIsNone(prepare_run(root,request));self.assertTrue((root/'attempts'/'interrupted').exists())
 def test_completed_result_recovers_missing_manifest_and_preserves_scratch(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);request={'commit':'a'};prepare_run(root,request);result={'request':request,'status':'passed'};write_json(root/'result.json',result);(root/'checksums.tmp').write_text('{partial')
   self.assertEqual(prepare_run(root,request),result);hashes=json.loads((root/'checksums.json').read_text());self.assertNotIn('checksums.tmp',hashes)
   for name,sha in hashes.items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),sha)
   self.assertEqual(prepare_run(root,request),result)
 def test_rejects_foreign_request_and_unknown_output(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);prepare_run(root,{'commit':'a'})
   with self.assertRaisesRegex(ValueError,'Different request'):prepare_run(root,{'commit':'b'})
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'unknown-result.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'Unknown existing'):prepare_run(root,{'commit':'a'})
 def test_completed_result_tampering_is_not_resealed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);request={'commit':'a'};prepare_run(root,request);write_json(root/'result.json',{'request':request,'status':'passed'});seal(root);write_json(root/'result.json',{'request':request,'status':'changed'})
   with self.assertRaisesRegex(ValueError,'sealed probe artifact changed'):prepare_run(root,request)
if __name__=='__main__':unittest.main()
