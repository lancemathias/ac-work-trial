"""Compare batch-4 against fresh batch-1 and preserved completed v2 cache rows."""
import json,time,pathlib,dataclasses,hashlib
import torch
from safetensors.torch import load_file
from cache_hidden_runtime import collate_records,EncodedRecord,EncodedQuestion,JointSchemaHead,write_json,digest
from backbone_cache_metadata import AttentionRecorder

def run(snapshot,cache,out):
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 cfg=snapshot['config']['config'];cache=pathlib.Path(cache);out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 progress=json.loads((cache/'progress.json').read_text());index=progress['index'];assert index
 torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 processor=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir=cfg['model_cache']);tok=processor.tokenizer
 backbone=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir=cfg['model_cache'],dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'}).eval();backbone.requires_grad_(False)
 model=backbone.model.language_model if hasattr(backbone.model,'language_model') else backbone.model;dim=backbone.get_output_embeddings().weight.shape[1]
 torch.manual_seed(cfg['seed']);head=JointSchemaHead(hidden_size=dim,**cfg['head']).float().cuda().eval()
 # The same deterministic parity-test head as extraction, not a trained checkpoint.
 records=[];shard_cache={}
 for s in progress['shards']:
  assert digest(cache/s['path'])==s['sha256']
 for row in index:
  name=row['shard']
  if name not in shard_cache:shard_cache[name]=load_file(str(cache/name))
  t=shard_cache[name];start=row['token_offset'];end=start+row['token_count'];ids=t['input_ids'][start:end]
  q=tuple(EncodedQuestion(x['question_id'],x['question_type'],tuple(x['question_span']),tuple(tuple(s) for s in x['option_spans']),tuple(x['option_ids'])) for x in row['questions'])
  records.append((row,EncodedRecord(tuple(ids.tolist()),q,row['id']),t['hidden_states'][start:end]))
 records.sort(key=lambda x:(len(x[1].input_ids),x[0]['id']))
 # Warm both shapes outside timing.
 def forward(batch):
  with torch.no_grad():return model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
 for group in [records[:1],records[:4]]:
  b=collate_records([r[1] for r in group],tok.pad_token_id,torch.device('cuda'));forward(b)
 torch.cuda.synchronize();results=[];timings={1:0.,4:0.};peak={1:0,4:0};valid_tokens=0;padded_tokens=0;trace=None
 def diff(a,b):
  d=(a.float()-b.float()).abs();return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rms':float(d.square().mean().sqrt())}
 for begin in range(0,len(records),4):
  group=records[begin:begin+4];singles=[]
  for row,encoded,cached in group:
   b=collate_records([encoded],tok.pad_token_id,torch.device('cuda'));torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();h=forward(b);torch.cuda.synchronize();timings[1]+=time.perf_counter()-start;peak[1]=max(peak[1],torch.cuda.max_memory_allocated());singles.append(h[0].cpu())
  b=collate_records([r[1] for r in group],tok.pad_token_id,torch.device('cuda'));torch.cuda.reset_peak_memory_stats();rec=AttentionRecorder()
  if trace is None:rec.install();rec.begin()
  torch.cuda.synchronize();start=time.perf_counter();h4=forward(b);torch.cuda.synchronize();timings[4]+=time.perf_counter()-start;peak[4]=max(peak[4],torch.cuda.max_memory_allocated())
  if trace is None:trace=rec.end();rec.restore()
  padded_tokens+=b['input_ids'].numel()
  for i,(row,encoded,cached) in enumerate(group):
   n=len(encoded.input_ids);valid_tokens+=n;h1=singles[i].cuda();h=h4[i,:n];one=collate_records([encoded],tok.pad_token_id,torch.device('cuda'))
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    logits1=head(h1.unsqueeze(0),one['input_ids'],one['attention_mask'],one['records'],backbone.get_output_embeddings().weight)[0]
    logits4=head(h.unsqueeze(0),one['input_ids'],one['attention_mask'],one['records'],backbone.get_output_embeddings().weight)[0]
    # Backbone vocabulary logits at each question's final option token.
    positions=[end-1 for q in encoded.questions for _,end in q.option_spans]
    vocab1=backbone.get_output_embeddings()(h1[positions]);vocab4=backbone.get_output_embeddings()(h[positions])
   qdiff=[]
   for a,z in zip(logits1,logits4):
    qdiff.append({'logits':diff(a,z),'probability_max_abs':float((a.float().softmax(-1)-z.float().softmax(-1)).abs().max()),'prediction_changed':bool(a.argmax()!=z.argmax())})
   results.append({'id':row['id'],'source':row['source'],'tokens':n,'padding_tokens':h4.shape[1]-n,'hidden_batch4_vs_fresh1':diff(h,h1),'hidden_fresh1_vs_saved1':diff(h1,cached.cuda()),'backbone_vocabulary_logits':diff(vocab4,vocab1),'questions':qdiff})
  write_json(out/'progress.json',{'records':len(results),'total':len(records),'forward_seconds':timings})
  print(json.dumps({'compared':len(results),'total':len(records)}),flush=True)
 report={'status':'complete','records':len(results),'dataset_version':'v2','source_cache_identity':progress['identity'],'source_progress_sha256':digest(cache/'progress.json'),'batching':'Length-sorted groups of four, right padding, original collator; mask 1 valid / 0 padding; position_ids omitted as in original; eval/no_grad/use_cache=False. Final remainder may be <4.','comparison':'Fresh batch-1 and batch-4 on same GPU/runtime; original saved batch-1 as secondary control.','head':'Deterministic randomly initialized parity-test head, seed20261014 and source-run head config; NOT trained model weights. Vocabulary logits from actual frozen backbone also compared.','forward_seconds':timings,'valid_tokens':valid_tokens,'padded_tokens_batch4':padded_tokens,'tokens_per_second':{str(k):valid_tokens/v for k,v in timings.items()},'speedup_forward_only':timings[1]/timings[4],'peak_allocated_bytes':peak,'hidden_max_abs':max(r['hidden_batch4_vs_fresh1']['max_abs'] for r in results),'vocabulary_logits_max_abs':max(r['backbone_vocabulary_logits']['max_abs'] for r in results),'head_logits_max_abs':max(q['logits']['max_abs'] for r in results for q in r['questions']),'head_probability_max_abs':max(q['probability_max_abs'] for r in results for q in r['questions']),'head_prediction_changes':sum(q['prediction_changed'] for r in results for q in r['questions']),'observed_first_batch4_attention':trace,'torch':str(torch.__version__),'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(),'per_record':results}
 write_json(out/'comparison.json',report);return {k:v for k,v in report.items() if k not in ['per_record','observed_first_batch4_attention','source_cache_identity']}
