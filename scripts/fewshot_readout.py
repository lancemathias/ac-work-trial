"""Fixed seven approved training demonstrations, unchanged native query payload."""
from collections.abc import Mapping

def gold_option(row):
 if row['question']['type']=='noul':
  if type(row['label']) is not bool:raise ValueError('Boolean training label required')
  return 'true' if row['label'] else 'false'
 return row['label']

def messages(state,question,demonstrations):
 from pretrained_readout import messages as native_messages,CODES
 if len(demonstrations)!=7 or len({r['id'] for r in demonstrations})!=7:raise ValueError('Exactly seven distinct fixed training demonstrations required')
 base,options=native_messages(state,question);turns=[base[0]]
 for row in demonstrations:
  if row.get('evaluation_only') is not False:raise ValueError('Heldout/evaluation demonstration forbidden')
  example,keys=native_messages(row['state'],row['question'])
  if len(example)!=2 or example[0]!=base[0]:raise ValueError('Fixed native system prompt required')
  answer=CODES[keys.index(gold_option(row))]
  turns.extend([example[1],{'role':'assistant','content':answer}])
 turns.extend(base[1:]);return turns,options

def encode(tokenizer,state,question,demonstrations):
 from pretrained_readout import encode as native_encode
 native=native_encode(tokenizer,state,question,maximum=8192);turns,options=messages(state,question,demonstrations)
 if options!=native['option_ids']:raise ValueError('Query option semantics changed')
 ids=tokenizer.apply_chat_template(turns,tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if isinstance(ids,Mapping):ids=ids['input_ids']
 if hasattr(ids,'tolist'):ids=ids.tolist()
 if not isinstance(ids,list) or not ids or any(type(x)is not int for x in ids):raise ValueError('Expected flat token IDs')
 if len(ids)>12288 or len(ids)-len(native['input_ids'])>4096:raise ValueError('Fewshot context exceeds preregistered cap; no truncation')
 return {'input_ids':ids,'option_ids':options,'code_token_ids':native['code_token_ids']}
