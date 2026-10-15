import copy,hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cache_resume import compatible_identity,require_exact_replay
class ResumeTests(unittest.TestCase):
 def fixture(self,uuid='one',driver='610',mask=False):
  raw=json.dumps({'nvidia_smi':f'A100, GPU-{uuid}, {driver}, 81920 MiB','mask':mask}).encode()
  return {'backbone_metadata':{'backend_settings_sha256':hashlib.sha256(raw).hexdigest(),'model_config_sha256':'model'},'dataset_sha256':'data','seed':7},raw
 def test_uuid_only_difference_accepted_with_original_identity(self):
  old,a=self.fixture();new,b=self.fixture(uuid='two');self.assertEqual(compatible_identity(old,new,a,b),old)
 def test_semantic_differences_and_metadata_tampering_rejected(self):
  old,a=self.fixture()
  for kwargs in [{'driver':'other'},{'mask':True}]:
   new,b=self.fixture(**kwargs)
   with self.assertRaisesRegex(ValueError,'backend differs'):compatible_identity(old,new,a,b)
  new,b=self.fixture(uuid='two');new['seed']=8
  with self.assertRaisesRegex(ValueError,'computation identity'):compatible_identity(old,new,a,b)
  with self.assertRaisesRegex(ValueError,'checksum'):compatible_identity(old,old,a,b)
 def test_live_replay_rejects_changed_features(self):
  import torch
  x=torch.tensor([1.,2.],dtype=torch.bfloat16);require_exact_replay(x,x.clone())
  with self.assertRaisesRegex(ValueError,'parity'):require_exact_replay(x,x+1)
if __name__=='__main__':unittest.main()
