import unittest,copy
from types import SimpleNamespace
from bounded_thinking_readout import parse_thought,final_input,prepare,predict
class Tokenizer:
 pad_token_id=0
 def __call__(self,text,add_special_tokens=False):
  return SimpleNamespace(input_ids=[126] if text=='</think>' else [4] if text=='\n\n' else [ord(text)-ord('A')+1])
 def apply_chat_template(self,turns,tokenize=True,add_generation_prompt=True,enable_thinking=False):return [5,6,7] if enable_thinking else [5,6]
class ThinkingTests(unittest.TestCase):
 def settings(self):return dict(original_native_prompt_cap=32,thinking_prompt_cap=40,reasoning_token_budget=3,closure_token_allowance=8,final_context_cap=48,batch_size=2)
 def test_closure_stops_and_budget_are_distinguished_without_dropping_content(self):
  a=parse_thought([8,126,0],126,[2],3);self.assertEqual(a['tokens'],[8]);self.assertEqual(a['stop_kind'],'closed');self.assertEqual(final_input([5,6],a,126,[4],self.settings()),[5,6,8,126,4])
  b=parse_thought([8,9,10],126,[2],3);self.assertEqual(b['stop_kind'],'budget');self.assertEqual(final_input([5],b,126,[4],self.settings()),[5,8,9,10,4,126,4])
  self.assertEqual(parse_thought([8,2,0],126,[2],3)['stop_kind'],'model_end')
  with self.assertRaises(ValueError):parse_thought([1,2,3,4],126,[2],3)
 def test_gold_cannot_enter_thinking_inputs_and_calibration_target_stays_semantic(self):
  row={'id':'x','state':'public','question':{'type':'noul','instructions':'True?'},'label':True,'source':{'audit':'private'}};before=prepare(Tokenizer(),[row],self.settings());row['label']=False;row['source']['audit']='changed';self.assertEqual(before,prepare(Tokenizer(),[row],self.settings()));self.assertEqual(set(before[0]),{'id','type','encoding'})
  train={'id':'x','source':'allowed','request':{'state':'public','questions':{'q':row['question']}},'targets':{'q':'false'}};cal=prepare(Tokenizer(),[train],self.settings(),True);self.assertEqual(cal[0]['target_index'],1)
 def test_real_tiny_qwen_cached_generation_and_full_readout_batch_contract(self):
  import torch
  from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
  torch.set_num_threads(1);torch.manual_seed(47)
  cfg=Qwen3_5TextConfig(vocab_size=129,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=4,layer_types=['linear_attention','full_attention'],bos_token_id=1,eos_token_id=2,pad_token_id=0)
  model=Qwen3_5ForCausalLM(cfg).requires_grad_(False).eval();rows=[{'id':'short','type':'choice','encoding':{'input_ids':[5,6],'option_ids':['a','b'],'code_token_ids':[1,2]}},{'id':'long','type':'choice','encoding':{'input_ids':[8,9,10,11],'option_ids':['a','b'],'code_token_ids':[1,2]}}];settings=self.settings();batched=predict(model,Tokenizer(),rows,settings);settings=copy.deepcopy(settings);settings['batch_size']=1;single=predict(model,Tokenizer(),rows,settings)
  self.assertEqual({x['id'] for x in batched},{'short','long'})
  for a,b in zip(batched,single):
   self.assertEqual(a['reasoning_token_ids'],b['reasoning_token_ids']);self.assertEqual(a['expanded_input_sha256'],b['expanded_input_sha256']);torch.testing.assert_close(torch.tensor(a['logits']),torch.tensor(b['logits']),rtol=1e-4,atol=1e-5);self.assertLessEqual(a['generated_tokens'],3)
  self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
