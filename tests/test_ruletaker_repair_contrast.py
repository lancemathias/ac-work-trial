import copy,unittest
from verify_ruletaker_repair_contrast import parent_checks
class RepairContrast(unittest.TestCase):
 def fixture(self):
  keys=['backbone_id','backbone_revision','encoder_sha256','adapter_sha256','dtype','hidden_layer','attention_implementation','extraction_batch_size','padding','max_input_tokens','no_truncation','torch','transformers','hidden_size','head_config','tf32','cudnn_sdp','attention_mode','attention_hook_sha256','optimized_kernel_runtime']
  a={'feature_cache':{'identity':dict.fromkeys(keys,'fixed')}};a['feature_cache']['identity']['parent_manifest_sha256']=['base','old-union'];b=copy.deepcopy(a);b['feature_cache']['identity']['parent_manifest_sha256']=['base','fresh250'];b['config']={'dataset':{'sha256':'full'}}
  return a,b,{'status':'complete','cached_head_backward_verified':True,'identity':{'record_count':7928}},{'status':'complete','cached_head_backward_verified':True,'identity':{'record_count':250,'dataset_sha256':'additions'}},{'additions_sha256':'additions','full_dataset_sha256':'full'}
 def test_only_explicit_fresh_parent_allowed(self):self.assertTrue(all(parent_checks(*self.fixture()).values()))
 def test_old_feature_reuse_rejected(self):
  args=self.fixture();args[1]['feature_cache']['identity']['parent_manifest_sha256'][1]='old-union';self.assertFalse(parent_checks(*args)['fresh_addition_parent'])
 def test_base_change_rejected(self):
  args=self.fixture();args[1]['feature_cache']['identity']['parent_manifest_sha256'][0]='other';self.assertFalse(parent_checks(*args)['same_base_parent'])
 def test_kernel_change_rejected(self):
  args=self.fixture();args[1]['feature_cache']['identity']['optimized_kernel_runtime']='different';self.assertFalse(parent_checks(*args)['frozen_extraction_regime_exact'])
 def test_wrong_requests_rejected(self):
  args=self.fixture();args[3]['identity']['dataset_sha256']='wrong';self.assertFalse(parent_checks(*args)['fresh_request_bytes_match_preparation'])
if __name__=='__main__':unittest.main()
