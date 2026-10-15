import json,unittest
from pretrained_readout import messages,encode
from evaluate_pretrained_readout import prepare_decisions
from test_pretrained_readout import Tokenizer
class OrderEncodingTests(unittest.TestCase):
 def test_reversal_only_reassigns_codes_and_keeps_semantics(self):
  for q in [{'type':'noul','instructions':'True?'},{'type':'choice','instructions':'Choose','criteria':{'z':'last','a':'first','m':'middle'}},{'type':'score','instructions':'Rate','criteria':['low','mid','high']}]:
   native,ids=messages('same state',q);reverse,rids=messages('same state',q,'reverse')
   self.assertEqual(rids,list(reversed(ids)));self.assertEqual(native[0],reverse[0])
   self.assertEqual(native[1]['content'].split('\n\nOPTIONS:')[0],reverse[1]['content'].split('\n\nOPTIONS:')[0])
   def meanings(turns):return {x['option_id']:x['meaning'] for x in json.loads(turns[1]['content'].split('\n\nOPTIONS:\n')[1])}
   self.assertEqual(meanings(native),meanings(reverse))
   n=encode(Tokenizer(),'same state',q);r=encode(Tokenizer(),'same state',q,option_order='reverse')
   self.assertEqual(n['code_token_ids'],r['code_token_ids'])
 def test_reversed_training_target_is_semantically_remapped(self):
  row={'id':'x','source':'permitted','request':{'state':'public','questions':{'q':{'type':'noul','instructions':'True?'}}},'targets':{'q':'true'}}
  a=prepare_decisions(Tokenizer(),[row],True)[0];b=prepare_decisions(Tokenizer(),[row],True,option_order='reverse')[0]
  self.assertEqual(a['target_index'],0);self.assertEqual(b['target_index'],1)
  self.assertEqual(a['encoding']['option_ids'][a['target_index']],b['encoding']['option_ids'][b['target_index']])
 def test_reverse_test_encoding_excludes_gold(self):
  row={'id':'x','state':'public','question':{'type':'noul','instructions':'True?'},'label':'private','source':{'audit':'secret'}}
  a=prepare_decisions(Tokenizer(),[row],option_order='reverse');row['label']='different';row['source']['audit']='changed'
  self.assertEqual(a,prepare_decisions(Tokenizer(),[row],option_order='reverse'));self.assertEqual(set(a[0]),{'id','type','encoding'})
 def test_unknown_order_rejected(self):
  with self.assertRaises(ValueError):messages('',{'type':'noul','instructions':'True?'},'random')
