"""Fixed128-token native thinking, then semantic answer-code probabilities."""
import hashlib,json,math,time
from collections.abc import Mapping

def flat_ids(value):
 if isinstance(value,Mapping):value=value['input_ids']
 if hasattr(value,'tolist'):value=value.tolist()
 if not isinstance(value,list) or not value or any(type(x)is not int for x in value):raise ValueError('Expected one flat token sequence')
 return value

def encode(tokenizer,state,question,settings):
 from pretrained_readout import messages,encode as native_encode
 native=native_encode(tokenizer,state,question,maximum=settings['original_native_prompt_cap']);turns,options=messages(state,question)
 ids=flat_ids(tokenizer.apply_chat_template(turns,tokenize=True,add_generation_prompt=True,enable_thinking=True))
 if len(ids)>settings['thinking_prompt_cap']:raise ValueError('Thinking prompt exceeds amended cap; no truncation')
 return dict(input_ids=ids,option_ids=options,code_token_ids=native['code_token_ids'])

def prepare(tokenizer,rows,settings,training_sources=False):
 result=[]
 for row in rows:
  state=row['request']['state'] if training_sources else row['state'];question=row['request']['questions']['q'] if training_sources else row['question'];e=encode(tokenizer,state,question,settings)
  item={'id':row['id'],'type':question['type'],'encoding':e}
  if training_sources:item.update(source=row['source'],target_index=e['option_ids'].index(row['targets']['q']))
  result.append(item)
 return result

def parse_thought(generated,closing,model_ends,budget):
 if len(generated)>budget:raise ValueError('Generation exceeded fixed reasoning budget')
 for i,token in enumerate(generated):
  if token==closing:return {'tokens':generated[:i],'stop_kind':'closed','generated_tokens':i+1}
  if token in model_ends:return {'tokens':generated[:i],'stop_kind':'model_end','generated_tokens':i+1}
 return {'tokens':generated,'stop_kind':'budget','generated_tokens':len(generated)}

def final_input(prefix,thought,closing,newlines,settings):
 suffix=([closing]+newlines) if thought['stop_kind']=='closed' else (newlines+[closing]+newlines)
 if len(suffix)>settings['closure_token_allowance']:raise ValueError('Closure overhead exceeds preregistered allowance')
 result=prefix+thought['tokens']+suffix
 if len(result)>settings['final_context_cap']:raise ValueError('Expanded context exceeds fixed cap; no truncation')
 return result

def predict(backbone,tokenizer,prepared,settings,emit=lambda x:None):
 import torch
 from pretrained_readout import selected_logits
 closing=tokenizer('</think>',add_special_tokens=False).input_ids
 if len(closing)!=1:raise ValueError('Closing-think tag must be one token')
 closing=closing[0];newlines=flat_ids(tokenizer('\n\n',add_special_tokens=False).input_ids)
 eos=getattr(backbone.generation_config,'eos_token_id',None);model_ends=([eos] if isinstance(eos,int) else list(eos or []));stops=list(dict.fromkeys([closing]+model_ends))
 if tokenizer.pad_token_id is None:raise ValueError('Tokenizer pad ID required')
 ordered=sorted(prepared,key=lambda x:(len(x['encoding']['input_ids']),x['id']));device=next(backbone.parameters()).device;frozen=[(p,p._version) for p in backbone.parameters()];result=[];started=time.monotonic();batch_size=settings['batch_size']
 if backbone.training or any(p.requires_grad or p.grad is not None for p,_ in frozen):raise ValueError('Require frozen evaluation model')
 for begin in range(0,len(ordered),batch_size):
  group=ordered[begin:begin+batch_size];width=max(len(x['encoding']['input_ids']) for x in group);ids=torch.full((len(group),width),tokenizer.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
  for i,row in enumerate(group):
   tokens=row['encoding']['input_ids'];ids[i,-len(tokens):]=torch.tensor(tokens,device=device);mask[i,-len(tokens):]=1
  tick=time.monotonic()
  with torch.no_grad():generation=backbone.generate(input_ids=ids,attention_mask=mask,max_new_tokens=settings['reasoning_token_budget'],do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=False,logits_to_keep=1,eos_token_id=stops,pad_token_id=tokenizer.pad_token_id)
  if device.type=='cuda':torch.cuda.synchronize()
  reasoning_ms=1000*(time.monotonic()-tick)/len(group);generated=generation.sequences[:,width:].cpu().tolist();thoughts=[parse_thought(x,closing,model_ends,settings['reasoning_token_budget']) for x in generated]
  # Drop generated KV state before the final full prefill; never mix padded cache states.
  del generation,ids,mask
  finals=[final_input(row['encoding']['input_ids'],t,closing,newlines,settings) for row,t in zip(group,thoughts)];lengths=[len(x) for x in finals];width=max(lengths);ids=torch.full((len(group),width),tokenizer.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
  for i,tokens in enumerate(finals):ids[i,:len(tokens)]=torch.tensor(tokens,device=device);mask[i,:len(tokens)]=1
  codes=max((x['encoding']['code_token_ids'] for x in group),key=len)
  if any(x['encoding']['code_token_ids']!=codes[:len(x['encoding']['code_token_ids'])] for x in group):raise ValueError('Inconsistent answer letter tokens')
  text=backbone.model
  if hasattr(text,'language_model'):text=text.language_model
  tick=time.monotonic()
  with torch.no_grad():
   hidden=text(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state;last=hidden[torch.arange(len(group),device=device),torch.tensor(lengths,device=device)-1];logits=selected_logits(backbone,last,codes).cpu()
  if device.type=='cuda':torch.cuda.synchronize()
  readout_ms=1000*(time.monotonic()-tick)/len(group)
  for i,(row,tokens,thought) in enumerate(zip(group,finals,thoughts)):
   values=logits[i,:len(row['encoding']['option_ids'])]
   if not torch.isfinite(values).all():raise ValueError('Nonfinite bounded-thinking answer logits')
   item={k:v for k,v in row.items() if k!='encoding'};item.update(option_ids=row['encoding']['option_ids'],logits=values.tolist(),input_tokens=len(row['encoding']['input_ids']),expanded_tokens=len(tokens),reasoning_token_ids=thought['tokens'],reasoning_stop=thought['stop_kind'],generated_tokens=thought['generated_tokens'],reasoning_ms=reasoning_ms,readout_ms=readout_ms,latency_ms=reasoning_ms+readout_ms,expanded_input_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest());result.append(item);emit(item)
  del hidden,last,logits,ids,mask
  if begin//batch_size%25==0:print(json.dumps({'event':'bounded_thinking_progress','records':len(result),'total':len(ordered),'seconds':time.monotonic()-started}),flush=True)
 if any(p.requires_grad or p.grad is not None or p._version!=v for p,v in frozen):raise RuntimeError('Thinking inference modified frozen weights')
 return result
