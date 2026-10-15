"""Behavioral tests: suffix visibility, padding isolation and frozen weights."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
from backbone_attention import install_attention_mode,BIDIRECTIONAL
class AttentionTests(unittest.TestCase):
 def test_suffix_visibility_padding_and_frozen_parameters(self):
  torch.set_num_threads(1);torch.manual_seed(13)
  cfg=Qwen3_5TextConfig(vocab_size=129,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=4,layer_types=['linear_attention','full_attention','linear_attention'])
  cfg._attn_implementation='sdpa';model=Qwen3_5ForCausalLM(cfg).requires_grad_(False).eval()
  weights={k:v.clone() for k,v in model.state_dict().items()}
  ids=torch.tensor([[1,2,3,4,5,6,0,0],[11,12,13,14,15,16,17,18]]);mask=torch.tensor([[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1]])
  def forward(tokens):
   with torch.no_grad():return model.model(input_ids=tokens,attention_mask=mask,use_cache=False).last_hidden_state
  suffix=ids.clone();suffix[:,-4:-2]+=30
  native=forward(ids);native_changed=forward(suffix)
  torch.testing.assert_close(native[:,:4],native_changed[:,:4],atol=0,rtol=0)
  install_attention_mode(model,BIDIRECTIONAL);bidirectional=forward(ids);changed=forward(suffix)
  self.assertGreater(float((bidirectional[:,:4]-changed[:,:4]).abs().max()),1e-5)
  padded=ids.clone();padded[0,6:]=torch.tensor([100,101])
  torch.testing.assert_close(bidirectional[0,:6],forward(padded)[0,:6],atol=0,rtol=0)
  for k,v in model.state_dict().items():torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
  self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
 def test_unknown_mode_rejected(self):
  with self.assertRaises(ValueError):install_attention_mode(None,'unknown')
if __name__=='__main__':unittest.main()
