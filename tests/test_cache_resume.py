import copy,hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cache_resume import compatible_identity,require_exact_replay,require_extraction_phase
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
 def test_only_diagnosed_manual_stop_or_extraction_phase_allowed(self):
  marker={'previous':{'phase':'extracting_frozen_features','run_id':'x'},'recovery_commit':'commit'}
  state={'phase':'failed','manual_recovery':True,'error':'RemoteError','message':'','run_id':'x','commit':'commit'}
  require_extraction_phase(state,marker)
  for field,value in [('error','ValueError'),('commit','wrong'),('phase','training_and_benchmark')]:
   changed=dict(state);changed[field]=value
   with self.assertRaisesRegex(ValueError,'diagnosed'):require_extraction_phase(changed,marker)
  with self.assertRaisesRegex(ValueError,'diagnosed'):require_extraction_phase(state,None)
 def test_only_native_legacy_identity_migration_allowed(self):
  old,a=self.fixture();new=copy.deepcopy(old);new.update(attention_mode='native_causal',attention_hook_sha256='hook')
  self.assertEqual(compatible_identity(old,new,a,a),old)
  new['attention_mode']='bidirectional_full_attention'
  with self.assertRaisesRegex(ValueError,'Legacy cache'):compatible_identity(old,new,a,a)
 def test_live_replay_rejects_changed_features(self):
  import torch
  x=torch.tensor([1.,2.],dtype=torch.bfloat16);require_exact_replay(x,x.clone())
  with self.assertRaisesRegex(ValueError,'parity'):require_exact_replay(x,x+1)
if __name__=='__main__':unittest.main()
