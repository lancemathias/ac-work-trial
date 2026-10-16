import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from types import SimpleNamespace
from head_runtime import configure_head,attention_context
from train_joint import forward_frozen

class HeadRuntimeTests(unittest.TestCase):
 def test_math_scope_does_not_reach_backbone_and_flags_restore(self):
  def flags():return (torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.math_sdp_enabled())
  before=flags();seen=[]
  class Text:
   def __call__(self,**kwargs):seen.append(('backbone',flags()));return SimpleNamespace(last_hidden_state=torch.ones(1,2,4))
  class Backbone:
   model=Text()
   def eval(self):return self
   def get_output_embeddings(self):return SimpleNamespace(weight=torch.ones(8,4))
  class Head:
   def __call__(self,*args):seen.append(('head',flags()));return args[0].sum()
  head=configure_head(Head(),{'head_sdpa_backend':'math'});batch={'input_ids':torch.ones(1,2,dtype=torch.long),'attention_mask':torch.ones(1,2),'records':[]}
  forward_frozen(Backbone(),head,batch)
  self.assertEqual(seen[0],('backbone',before));self.assertEqual(seen[1],('head',(False,False,True)));self.assertEqual(flags(),before)
 def test_default_auto_is_unchanged_and_invalid_mode_rejected(self):
  head=configure_head(SimpleNamespace(),{});before=torch.backends.cuda.flash_sdp_enabled()
  with attention_context(head):self.assertEqual(torch.backends.cuda.flash_sdp_enabled(),before)
  with self.assertRaises(ValueError):configure_head(head,{'head_sdpa_backend':'unknown'})
 def test_math_forward_backward_repeats_on_real_head_cpu_fixture(self):
  from vendor.clef_joint_schema import JointSchemaHead,encode_record,collate_records
  from test_training import CharacterTokenizer,TinyBackbone
  torch.set_num_threads(1);torch.manual_seed(7);backbone=TinyBackbone().requires_grad_(False).eval()
  request={'id':'x','state':'Evidence is present.','questions':{'q':{'type':'choice','instructions':'Choose.','criteria':{'a':'present','b':'absent'}}}}
  encoded=encode_record(CharacterTokenizer(),request,max_length=8192);batch=collate_records([encoded],0,torch.device('cpu'));states=[]
  for _ in range(2):
   torch.manual_seed(13);head=configure_head(JointSchemaHead(hidden_size=8,width=8,routing_layers=1,layers=1,heads=2,feedforward=16,dropout=0),{'head_sdpa_backend':'math'})
   values=forward_frozen(backbone,head,batch)[0][0];values.square().sum().backward();states.append((values.detach().clone(),[p.grad.clone() for p in head.parameters() if p.grad is not None]))
  torch.testing.assert_close(states[0][0],states[1][0],atol=0,rtol=0)
  for a,b in zip(states[0][1],states[1][1]):torch.testing.assert_close(a,b,atol=0,rtol=0)
 def test_cached_and_live_math_training_updates_are_identical(self):
  from unittest.mock import patch
  import test_cached_training as cached
  original=cached.JointSchemaHead
  def constructor(*args,**kwargs):return configure_head(original(*args,**kwargs),{'head_sdpa_backend':'math'})
  with patch.object(cached,'JointSchemaHead',side_effect=constructor):
   cached.CachedTrainingTests('test_same_features_produce_identical_head_updates').test_same_features_produce_identical_head_updates()
