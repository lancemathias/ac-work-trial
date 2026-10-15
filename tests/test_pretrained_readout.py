"""Gold exclusion, code encoding, typed semantics and pretrained score parity."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pretrained_readout import messages,encode,selected_logits
class Tokenizer:
 pad_token_id=0
 def __call__(self,text,add_special_tokens=False):return SimpleNamespace(input_ids=[int(text)+1])
 def apply_chat_template(self,turns,**kwargs):
  assert kwargs=={'tokenize':True,'add_generation_prompt':True,'enable_thinking':False}
  self.turns=turns;return [1,2,3]
class ReadoutTests(unittest.TestCase):
 def test_prompt_preserves_semantics_and_has_no_gold_channel(self):
  q={'type':'choice','instructions':'Which answer?','criteria':{'z':'last','a':'first'}}
  t=Tokenizer();e=encode(t,'public state',q)
  self.assertEqual(e['option_ids'],['a','z']);self.assertEqual(e['code_token_ids'],[1,2]);self.assertEqual(e['input_ids'],[1,2,3])
  content=t.turns[1]['content'];self.assertIn('public state',content);self.assertIn('first',content);self.assertNotIn('target',content);self.assertNotIn('label',content)
  self.assertEqual(messages({}, {'type':'noul','instructions':'Is it true?'})[1],['true','false'])
  self.assertEqual(messages({}, {'type':'score','instructions':'Rate','criteria':['low','high']})[1],['0','1'])
 def test_chat_template_mapping_returns_flat_ids(self):
  class MappingTokenizer(Tokenizer):
   def apply_chat_template(self,*args,**kwargs):return {'input_ids':[1,2,3,4],'attention_mask':[1,1,1,1]}
  q={'type':'noul','instructions':'True?'}
  self.assertEqual(encode(MappingTokenizer(),{},q)['input_ids'],[1,2,3,4])
  with self.assertRaisesRegex(ValueError,'no truncation'):encode(MappingTokenizer(),{},q,maximum=3)
 def test_multitoken_code_and_overlength_are_rejected(self):
  class Bad(Tokenizer):
   def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=[1,2])
  q={'type':'noul','instructions':'True?'}
  with self.assertRaisesRegex(ValueError,'one token'):encode(Bad(),{},q)
  with self.assertRaisesRegex(ValueError,'no truncation'):encode(Tokenizer(),{},q,maximum=2)
 def test_selected_scores_equal_full_pretrained_output(self):
  import torch
  torch.manual_seed(7);head=torch.nn.Linear(8,32,bias=True);backbone=SimpleNamespace(get_output_embeddings=lambda:head);hidden=torch.randn(3,8);codes=[2,5,11]
  actual=selected_logits(backbone,hidden,codes);expected=head(hidden)[:,codes]
  torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
if __name__=='__main__':unittest.main()
