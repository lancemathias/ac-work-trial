"""Resident reuse must preserve native outputs after a different attention mode."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
from resident_backbone import BackbonePool,get_model,get_processor
from backbone_attention import install_attention_mode,BIDIRECTIONAL
class ResidentTests(unittest.TestCase):
 def test_reuse_preserves_frozen_outputs_and_resets_attention(self):
  torch.set_num_threads(1);torch.manual_seed(13);loaded=[];processors=[]
  def loader(name,**kwargs):
   loaded.append((name,kwargs))
   cfg=Qwen3_5TextConfig(vocab_size=129,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=4,layer_types=['linear_attention','full_attention','linear_attention'])
   cfg._attn_implementation='sdpa';return Qwen3_5ForCausalLM(cfg)
  def processor(name,**kwargs):processors.append(name);return object()
  pool=BackbonePool(processor,loader);cfg={'backbone_id':'fixture','backbone_revision':'a'*40}
  ids=torch.tensor([[1,2,3,4,5,6]]);mask=torch.ones_like(ids)
  def hidden(model):
   with torch.no_grad():return model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
  with pool.activate():
   model=get_model(cfg);p=get_processor(cfg);native=hidden(model);weights={k:v.clone() for k,v in model.state_dict().items()}
   install_attention_mode(model,BIDIRECTIONAL);self.assertGreater(float((hidden(model)-native).abs().max()),1e-6)
   reused=get_model(cfg);self.assertIs(reused,model);self.assertIs(get_processor(cfg),p)
   torch.testing.assert_close(hidden(reused),native,atol=0,rtol=0)
   self.assertEqual(len(loaded),1);self.assertEqual(len(processors),1);self.assertTrue(loaded[0][1]['local_files_only'])
   for k,v in model.state_dict().items():torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
   self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
  with pool.activate():self.assertIs(get_model(cfg),model)
  self.assertEqual(pool.loads,1);self.assertEqual(pool.hits,2)
 def test_backbone_revision_change_does_not_reuse_old_weights(self):
  from torch import nn
  class Tiny(nn.Module):
   def __init__(self):
    super().__init__();self.weight=nn.Parameter(torch.ones(1));self.model=type('Text',(),{'layers':[]})()
  loaded=[]
  def loader(name,**kwargs):loaded.append(kwargs['revision']);return Tiny()
  pool=BackbonePool(model_loader=loader)
  a=pool.get_model({'backbone_id':'fixture','backbone_revision':'a'*40});b=pool.get_model({'backbone_id':'fixture','backbone_revision':'b'*40})
  self.assertIsNot(a,b);self.assertEqual(pool.loads,2);self.assertEqual(loaded,['a'*40,'b'*40])
if __name__=='__main__':unittest.main()
