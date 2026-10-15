"""Frozen full-token feature extraction and a head-compatible cache reader."""
import collections,dataclasses,hashlib,json,pathlib,time,sys,uuid
import torch
from safetensors.torch import load_file,save_file
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'hidden_cache_runtime'))
from training_data import adapt_training_point,canonical,digest
from vendor.clef_joint_schema import encode_record,collate_records,EncodedRecord,EncodedQuestion,JointSchemaHead

def write_json(p,value):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

class CachedEmbeddingLookup:
 """Frozen sparse lexical rows; implements the existing head's weight[token_ids]."""
 def __init__(self,ids,rows):self.ids=ids;self.rows=rows
 def __getitem__(self,token_ids):
  indices=torch.searchsorted(self.ids,token_ids)
  if torch.any(indices>=len(self.ids)) or not torch.equal(self.ids[indices],token_ids):raise ValueError('Uncached lexical token ID')
  return self.rows[indices]
 def detach(self):return self

class HiddenCache:
 """Read checked shards. Training remains responsible for split membership/order."""
 def __init__(self,directory,device='cuda',verify=True):
  self.root=pathlib.Path(directory);self.device=torch.device(device);self.manifest=json.loads((self.root/'manifest.json').read_text())
  if self.manifest['status']!='complete':raise ValueError('Incomplete feature cache')
  self.rows={r['id']:r for r in json.loads((self.root/'index.json').read_text())};self.loaded=None;self.shard=None
  if verify:
   for f in self.manifest['files']:
    if digest(self.root/f['path'])!=f['sha256']:raise ValueError('Cache checksum mismatch: '+f['path'])
  lexical=load_file(str(self.root/'output-embedding-rows.safetensors'),device=str(self.device));self.output_embeddings=CachedEmbeddingLookup(lexical['token_ids'],lexical['rows'])
 def get(self,rid):
  r=self.rows[rid]
  if self.shard!=r['shard']:self.loaded=load_file(str(self.root/r['shard']));self.shard=r['shard']
  start=r['token_offset'];end=start+r['token_count'];ids=self.loaded['input_ids'][start:end].to(self.device)
  hidden=self.loaded['hidden_states'][start:end].to(self.device)
  questions=tuple(EncodedQuestion(q['question_id'],q['question_type'],tuple(q['question_span']),tuple(tuple(s) for s in q['option_spans']),tuple(q['option_ids'])) for q in r['questions'])
  encoded=EncodedRecord(tuple(ids.cpu().tolist()),questions,rid)
  return {'hidden_states':hidden.unsqueeze(0),'input_ids':ids.unsqueeze(0),'attention_mask':torch.ones_like(ids).unsqueeze(0),'records':[encoded],'targets':r['targets'],'source':r['source'],'group_id':r['group_id']}
 def forward(self,head,rid,autocast=True):
  b=self.get(rid)
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast and self.device.type=='cuda'):
   return head(b['hidden_states'],b['input_ids'],b['attention_mask'],b['records'],self.output_embeddings)

def extract(run_snapshot,dataset,schema_path,out,checkpoint=lambda:None):
 from backbone_cache_metadata import collect_metadata,AttentionRecorder,tensor_info
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from jsonschema import Draft202012Validator
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 progress_path=out/'progress.json';old=json.loads(progress_path.read_text()) if progress_path.exists() else None
 previous_backend=(out/'backbone-metadata/backend-settings.json').read_bytes() if old else None
 cfg=run_snapshot['config']['config'];provenance=run_snapshot['config'];validator=Draft202012Validator(json.loads(pathlib.Path(schema_path).read_text()))
 assert digest(dataset)==cfg['dataset']['sha256'];assert digest(schema_path)==cfg['dataset']['schema_sha256']
 adapter=adapt_training_point
 if cfg['dataset'].get('format')=='normalized-benchmark-v1':
  from benchmark_training_data import adapt_training_point as adapter
  assert digest(HERE/'benchmark_training_data.py')==provenance['code_sha256']['scripts/benchmark_training_data.py']
 else:
  assert digest(HERE/'hidden_cache_runtime/training_data.py')==provenance['code_sha256']['scripts/training_data.py']
 assert digest(HERE/'hidden_cache_runtime/vendor/clef_joint_schema.py')==provenance['head_source_sha256']
 assert torch.__version__==provenance['torch'],(torch.__version__,provenance['torch'])
 assert __import__('transformers').__version__==provenance['transformers']
 torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(cfg.get('cpu_threads',8));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 processor=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir=cfg['model_cache']);tokenizer=processor.tokenizer
 points=[json.loads(line) for line in pathlib.Path(dataset).read_text().splitlines()];expected_count=cfg['dataset'].get('record_count',5000);assert len(points)==expected_count and len({p['id'] for p in points})==expected_count
 prepared=[];lexical_ids=set()
 for point in points:
  row=adapter(point,validator);encoded=encode_record(tokenizer,dict(row['request'],id=row['id']),max_length=10**9)
  if len(encoded.input_ids)>cfg['max_input_tokens']:raise ValueError(f"{row['id']}: exceeds max tokens; no truncation")
  prepared.append((row,encoded))
  for q in encoded.questions:
   for start,end in q.option_spans:lexical_ids.update(encoded.input_ids[start:end])
 prepared.sort(key=lambda pair:(len(pair[1].input_ids),pair[0]['id']))
 total_tokens=sum(len(e.input_ids) for _,e in prepared)
 print(json.dumps({'event':'preflight','records':len(prepared),'tokens':total_tokens,'maximum_tokens':max(len(e.input_ids) for _,e in prepared),'lexical_token_ids':len(lexical_ids)}),flush=True)
 backbone=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir=cfg['model_cache'],dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'})
 backbone.requires_grad_(False).eval()
 from backbone_attention import install_attention_mode
 mode=cfg.get('attention_mode','native_causal');install_attention_mode(backbone,mode)
 text_model=backbone.model
 if hasattr(text_model,'language_model'):text_model=text_model.language_model
 dim=backbone.get_output_embeddings().weight.shape[1]
 identity={'wandb_run_id':run_snapshot['id'],'wandb_url':run_snapshot['url'],'dataset_sha256':cfg['dataset']['sha256'],'schema_sha256':cfg['dataset']['schema_sha256'],'backbone_id':cfg['backbone_id'],'backbone_revision':cfg['backbone_revision'],'encoder_sha256':provenance['head_source_sha256'],'adapter_sha256':provenance['code_sha256'].get('scripts/benchmark_training_data.py',provenance['code_sha256']['scripts/training_data.py']),'dtype':'bfloat16','hidden_layer':'last_hidden_state after final model normalization','attention_implementation':'sdpa','extraction_batch_size':4,'batch_order':'ascending token length, then record ID','padding':'right; pad token ID from pinned tokenizer; mask 1 valid and 0 padded; padded states omitted from artifacts','max_input_tokens':cfg['max_input_tokens'],'no_truncation':True,'seed':cfg['seed'],'torch':torch.__version__,'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'hidden_size':dim,'total_tokens':total_tokens,'feature_bytes':total_tokens*dim*2,'record_count':len(prepared),'decision_count':sum(len(e.questions) for _,e in prepared),'head_config':cfg['head'],'tf32':False,'cudnn_sdp':False,'training_source_config':cfg}
 identity['attention_mode']=mode;identity['attention_hook_sha256']=digest(HERE/'backbone_attention.py')
 from kernel_runtime import optimized_runtime_identity
 optimized=optimized_runtime_identity()
 if optimized is not None:identity['optimized_kernel_runtime']=optimized
 metadata_root=out if not old else out/'resume-provenance'/uuid.uuid4().hex
 metadata_root.mkdir(parents=True,exist_ok=True)
 metadata=collect_metadata(backbone,processor,cfg,metadata_root);identity['backbone_metadata']=metadata
 write_json(metadata_root/'candidate-identity.json',identity)
 if old:
  from cache_resume import compatible_identity
  identity=compatible_identity(old['identity'],identity,previous_backend,(metadata_root/'backbone-metadata/backend-settings.json').read_bytes())
 process_path=metadata_root/'backbone-metadata/process.json';process=json.loads(process_path.read_text());process['input_attention_mask']='int64 [B,T], B<=4, right padding; 1 valid / 0 padded';process['batching']='ascending length then record ID; batches of 4, remainder allowed';write_json(process_path,process)
 write_json(out/'source-run.json',run_snapshot);write_json(out/'extraction-config.json',identity)
 recorder=AttentionRecorder();recorder.install()
 ids=torch.tensor(sorted(lexical_ids),dtype=torch.long);rows=backbone.get_output_embeddings().weight.detach()[ids.cuda()].cpu().contiguous();save_file({'token_ids':ids,'rows':rows},str(out/'output-embedding-rows.safetensors'))
 lookup=CachedEmbeddingLookup(ids.cuda(),rows.cuda());torch.manual_seed(cfg['seed']);head=JointSchemaHead(hidden_size=dim,**cfg['head']).float().cuda().eval()
 index=[];shards=[];start_time=time.monotonic();probe=[];count=0
 if old:
  from cache_resume import require_exact_replay
  replay_receipts=[]
  # Match original four-record padding/batching at both ends of the reused prefix.
  starts=sorted({0,max(0,((len(old['index'])-1)//4)*4)})
  for begin in starts:
   group=prepared[begin:begin+4];saved_rows=old['index'][begin:begin+4]
   if len(group)!=len(saved_rows):raise ValueError('Partial cache batch differs')
   padded=collate_records([e for _,e in group],tokenizer.pad_token_id,torch.device('cuda'))
   with torch.no_grad():live=text_model(input_ids=padded['input_ids'],attention_mask=padded['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
   for bi,((row,encoded),saved) in enumerate(zip(group,saved_rows,strict=True)):
    if row['id']!=saved['id']:raise ValueError('Resume record order differs')
    shard=next(x for x in old['shards'] if x['path']==saved['shard']);path=out/saved['shard']
    if digest(path)!=shard['sha256']:raise ValueError('Resume parity shard checksum differs')
    stored=load_file(str(path));offset=saved['token_offset'];n=saved['token_count']
    require_exact_replay(padded['input_ids'][bi,:n],stored['input_ids'][offset:offset+n])
    require_exact_replay(live[bi,:n],stored['hidden_states'][offset:offset+n])
    replay_receipts.append({'id':row['id'],'shard':saved['shard'],'bitwise_hidden_states_equal':True,'input_ids_equal':True})
   del live,padded,stored
  write_json(metadata_root/'resume-parity.json',{'status':'passed','records':replay_receipts,'policy':'Exact first and last cached four-record batches; all reused shards separately SHA256 verified.'});checkpoint()
  print(json.dumps({'event':'resume_parity_passed','records':len(replay_receipts),'cached_records':len(old['index'])}),flush=True)
 for begin in range(0,len(prepared),64):
  subset=prepared[begin:begin+64];shard_name=f'shards/{begin//64:04d}.safetensors';p=out/shard_name;p.parent.mkdir(exist_ok=True)
  cached=next((s for s in old['shards'] if s['path']==shard_name),None) if old else None
  if cached and p.exists() and digest(p)==cached['sha256']:
   shards.append(cached);index.extend(old['index'][begin:begin+len(subset)]);count+=len(subset);continue
  hidden_rows=[];input_rows=[];offset=0
  for batch_begin in range(0,len(subset),4):
   group=subset[batch_begin:batch_begin+4]
   padded=collate_records([e for _,e in group],tokenizer.pad_token_id,torch.device('cuda'))
   recorder.begin()
   with torch.no_grad():batched_hidden=text_model(input_ids=padded['input_ids'],attention_mask=padded['attention_mask'],use_cache=False,return_dict=True).last_hidden_state.detach()
   attention_trace=recorder.end()
   sdpa=[e for e in attention_trace if e['operation']=='scaled_dot_product_attention'];assert sdpa and all(e['is_causal'] or e['explicit_mask']['present'] for e in sdpa),'Unmasked noncausal SDPA detected'
   if not torch.isfinite(batched_hidden).all():raise ValueError('Non-finite batch states')
   causal_probe=None
   if not probe:
    changed=padded['input_ids'].clone();prefix=min(len(e.input_ids) for _,e in group)//2
    for bi,(_,e) in enumerate(group):changed[bi,prefix:len(e.input_ids)]=(changed[bi,prefix:len(e.input_ids)]+1)%backbone.config.text_config.vocab_size
    with torch.no_grad():future=text_model(input_ids=changed,attention_mask=padded['attention_mask'],use_cache=False,return_dict=True).last_hidden_state.detach()
    causal_probe=float((future[:,:prefix]-batched_hidden[:,:prefix]).abs().max())
    if mode=='native_causal':assert causal_probe==0,('Future tokens changed causal prefix',causal_probe)
    else:assert causal_probe>0,('Bidirectional ablation failed to expose suffix information',causal_probe)
   for bi,(row,encoded) in enumerate(group):
    n=len(encoded.input_ids);hidden=batched_hidden[bi:bi+1,:n]
    batch=collate_records([encoded],tokenizer.pad_token_id,torch.device('cuda'))
    if len(probe)<3:
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
      direct=head(hidden,batch['input_ids'],batch['attention_mask'],batch['records'],backbone.get_output_embeddings().weight.detach())[0]
      replayed=head(hidden.cpu().to('cuda'),batch['input_ids'],batch['attention_mask'],batch['records'],lookup)[0]
     error=max(float((a-b).abs().max()) for a,b in zip(direct,replayed));assert error==0,(row['id'],error)
     probe.append({'id':row['id'],'cached_vs_direct_head_max_abs_error':error,'future_suffix_perturbation_prefix_max_abs_error':causal_probe})
    h=hidden[0].cpu().contiguous();input_ids=batch['input_ids'][0].cpu().contiguous();hidden_rows.append(h);input_rows.append(input_ids)
    index.append({'id':row['id'],'source':row['source'],'group_id':row['group_id'],'targets':row['targets'],'shard':shard_name,'token_offset':offset,'token_count':n,'questions':[dataclasses.asdict(q) for q in encoded.questions],'attention_trace':attention_trace,'batch_record_ids':[r['id'] for r,_ in group],'batch_row_index':bi,'batch_padded_length':padded['input_ids'].shape[1],'input_attention_mask':tensor_info(padded['attention_mask']),'request_sha256':hashlib.sha256(canonical(row['request']).encode()).hexdigest()});offset+=n;count+=1
  temp=str(p)+'.tmp';save_file({'hidden_states':torch.cat(hidden_rows),'input_ids':torch.cat(input_rows)},temp);pathlib.Path(temp).replace(p)
  shards.append({'path':shard_name,'sha256':digest(p),'bytes':p.stat().st_size,'records':len(subset),'tokens':offset})
  write_json(progress_path,{'identity':identity,'index':index,'shards':shards,'probes':probe});checkpoint()
  print(json.dumps({'event':'cache_progress','records':count,'seconds':round(time.monotonic()-start_time,2),'feature_bytes_written':sum(s['bytes'] for s in shards)}),flush=True)
 recorder.restore()
 assert len(index)==expected_count and len({r['id'] for r in index})==expected_count
 write_json(out/'index.json',index)
 metadata_files=[{'path':str(p.relative_to(out)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted((out/'backbone-metadata').rglob('*')) if p.is_file()]
 resume_files=[{'path':str(p.relative_to(out)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted((out/'resume-provenance').rglob('*')) if p.is_file()]
 files=metadata_files+resume_files+shards+[{'path':name,'sha256':digest(out/name),'bytes':(out/name).stat().st_size} for name in ['index.json','output-embedding-rows.safetensors','source-run.json','extraction-config.json']]
 manifest={'status':'complete','identity':identity,'files':files,'parity_probes':probe or old['probes'],'total_bytes':sum(f['bytes'] for f in files),'source_counts':dict(collections.Counter(r['source'] for r in index)),'elapsed_seconds':time.monotonic()-start_time,'note':'Full per-token final BF16 states, not pooled vectors. Cached lexical output-embedding rows preserve the existing head. No backbone updates. No bitwise reproducibility guarantee across different CUDA/hardware/runtime.'}
 write_json(out/'manifest.json',manifest);checkpoint()
 cache=HiddenCache(out,verify=True)
 for row,_ in prepared[:3]:
  values=cache.forward(head,row['id'])[0];loss=sum(v.float().square().mean() for v in values);loss.backward();assert any(p.grad is not None for p in head.parameters());head.zero_grad(set_to_none=True)
 manifest['cached_head_backward_verified']=True;write_json(out/'manifest.json',manifest);checkpoint();return manifest
