import copy,unittest
from types import SimpleNamespace
from fewshot_readout import messages,encode
from pretrained_readout import messages as native_messages
class Tokenizer:
 def __init__(self,lengths=(20,200)):self.lengths=list(lengths);self.calls=[]
 def __call__(self,text,add_special_tokens=False):return SimpleNamespace(input_ids=[ord(text)])
 def apply_chat_template(self,turns,**kwargs):self.calls.append(copy.deepcopy(turns));return list(range(self.lengths[len(self.calls)-1]))
class FewshotContract(unittest.TestCase):
 def demos(self):return [{'id':'train'+str(i),'evaluation_only':False,'state':{'public':'training '+str(i)},'question':{'type':'noul','instructions':'Is it true?'},'label':i%2==0,'source':{'not_for_prompt':'audit-only'}} for i in range(7)]
 def test_native_query_and_system_preserved_correct_training_codes_only(self):
  state={'public':'query'};question={'type':'noul','instructions':'Is query supported?'};demos=self.demos();turns,options=messages(state,question,demos);base,keys=native_messages(state,question)
  self.assertEqual(turns[0],base[0]);self.assertEqual(turns[-1],base[-1]);self.assertEqual(options,keys);self.assertEqual([x['content'] for x in turns if x['role']=='assistant'],['A','B','A','B','A','B','A']);self.assertNotIn('audit-only',str(turns));self.assertEqual(len(turns),16)
  tok=Tokenizer();value=encode(tok,state,question,demos);self.assertEqual(value['option_ids'],['true','false']);self.assertEqual(value['code_token_ids'],[65,66]);self.assertEqual(len(value['input_ids']),200)
 def test_heldout_duplicate_or_wrong_demo_count_rejected(self):
  for kind in ['heldout','duplicate','count']:
   demos=self.demos()
   if kind=='heldout':demos[0]['evaluation_only']=True
   elif kind=='duplicate':demos[1]['id']=demos[0]['id']
   else:demos.pop()
   with self.assertRaises(ValueError):messages('query',{'type':'noul','instructions':'True?'},demos)
 def test_context_caps_reject_without_truncating(self):
  for lengths in [(8193,9000),(20,4200),(8192,12289)]:
   tok=Tokenizer(lengths)
   with self.assertRaises(ValueError):encode(tok,'query',{'type':'noul','instructions':'True?'},self.demos())
if __name__=='__main__':unittest.main()
