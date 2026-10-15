"""Same-A100 warm speed and dev-only numerical comparison, no parameter updates."""
import inspect,json,time,statistics,subprocess,importlib.metadata as metadata
from pathlib import Path

def actual_implementation(fn):
 while fn is not None:
  if inspect.isfunction(fn):
   values=inspect.getclosurevars(fn).nonlocals
   if 'implementation' in values:return values['implementation']
  fn=getattr(fn,'__wrapped__',None)
 raise ValueError('Cannot prove selected kernel implementation')

def run(run_id,commit,checkpoint):
 import torch
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 from safetensors.torch import load_file
 from training_data import load_splits,digest
 from train_joint import prepare,write_json
 from vendor.clef_joint_schema import JointSchemaHead,collate_records
 from backbone_attention import install_attention_mode
 out=Path('/results')/run_id;out.mkdir(exist_ok=False)
 names=['causal_conv1d_fn','causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']
 optimized={n:getattr(qwen,n) for n in names};fallback={n:inspect.unwrap(f) for n,f in optimized.items()}
 implementations={n:actual_implementation(f) for n,f in optimized.items()}
 paths={n:fn.__module__+'.'+fn.__name__ for n,fn in implementations.items()}
 if not paths['causal_conv1d_fn'].startswith('causal_conv1d.') or not paths['torch_chunk_gated_delta_rule'].startswith('fla.'):
  raise ValueError('Optimized packages did not become actual dispatched implementations: '+str(paths))
 parent=Path('/results/e4-accuracy-v3-seed20261014');cfg=json.loads((parent/'config.json').read_text())['config'];checks=json.loads((parent/'checksums.json').read_text())
 for name in ['config.json','best-head.safetensors','temperature.json']:
  if digest(parent/name)!=checks[name]:raise ValueError('Locked E4 artifact changed')
 splits,manifest=load_splits('/results/e4-accuracy-v3-seed20261014-splits/manifest.json')
 tokenizer=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 prepared=sorted(prepare(splits['dev'],tokenizer,8192),key=lambda pair:(len(pair[1].input_ids),pair[0]['id']))
 selected=[prepared[round(i*(len(prepared)-1)/31)] for i in range(32)]
 if len({r['id'] for r,_ in selected})!=32:raise ValueError('Quantiles must select unique inputs')
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'}).requires_grad_(False).eval()
 text=model.model
 if hasattr(text,'language_model'):text=text.language_model
 head=JointSchemaHead(hidden_size=model.get_output_embeddings().weight.shape[1],**cfg['head']).float().cuda();head.load_state_dict(load_file(parent/'best-head.safetensors',device='cuda'));head.requires_grad_(False).eval()
 frozen=[(p,p._version) for p in list(model.parameters())+list(head.parameters())];temperature=json.loads((parent/'temperature.json').read_text())['temperature']
 config={'commit':commit,'backbone_id':cfg['backbone_id'],'backbone_revision':cfg['backbone_revision'],'versions':{n:metadata.version(n) for n in ['torch','triton','transformers','fla-core','flash-linear-attention','causal-conv1d','einops']},'selected_implementations':paths,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'seed':cfg['seed'],'dtype':'bfloat16','sdpa':True,'tf32':False,'cudnn_sdp':False,'use_cache':False,'dev_sha256':manifest['splits']['dev']['points']['sha256'],'selection':'32 unique length quantiles, label-blind','selected_inputs':[{'id':r['id'],'tokens':len(e.input_ids)} for r,e in selected],'warm_repetitions':3,'protocol':'Same weights/device/inputs; alternate fallback and optimized order; cold first calls separate. CPU fixtures use reference ops, GPU test proves actual optimized dispatch.'}
 write_json(out/'config.json',config);checkpoint();speed=[];numerical=[]
 def choose(kind):
  for n,fn in (fallback if kind=='fallback' else optimized).items():setattr(qwen,n,fn)
 def batch(group):return collate_records([e for _,e in group],tokenizer.pad_token_id,torch.device('cuda'))
 def forward(b):
  with torch.no_grad():return text(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False,return_dict=True).last_hidden_state.detach()
 def timed(b,kind):
  choose(kind);torch.cuda.synchronize();start=time.perf_counter();h=forward(b);torch.cuda.synchronize();return time.perf_counter()-start,h
 def probabilities(h,b):
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):values=head(h,b['input_ids'],b['attention_mask'],b['records'],model.get_output_embeddings().weight.detach())
  return [z[0].float().cpu() for z in values]
 try:
  for mode in ['native_causal','bidirectional_full_attention']:
   handle=install_attention_mode(model,mode);mode_speed=[];mode_numeric=[]
   cases=[('single-short',selected[:1]),('single-median',selected[16:17]),('single-long',selected[-1:]),('batch4-median',selected[14:18]),('batch4-long',selected[-4:])]
   for label,group in cases:
    b=batch(group);cold={};times={kind:[] for kind in ['fallback','optimized']}
    for kind in times:
     seconds,h=timed(b,kind);cold[kind]=seconds;del h
    for repetition in range(3):
     for kind in (['fallback','optimized'] if repetition%2==0 else ['optimized','fallback']):
      seconds,h=timed(b,kind);times[kind].append(seconds);del h
    record={'attention_mode':mode,'case':label,'batch_size':len(group),'lengths':[len(e.input_ids) for _,e in group],'cold_first_seconds':cold,'warm_seconds':times,'warm_median_seconds':{k:statistics.median(v) for k,v in times.items()}}
    record['speedup']=record['warm_median_seconds']['fallback']/record['warm_median_seconds']['optimized'];speed.append(record);mode_speed.append(record)
    print(json.dumps({'event':'kernel_speed',**record}),flush=True);write_json(out/'speed.json',speed);checkpoint();del b
   for begin in range(0,32,4):
    group=selected[begin:begin+4];b=batch(group);choose('fallback');reference=forward(b);ref_logits=probabilities(reference,b);reference=reference.cpu();choose('optimized');candidate=forward(b);new_logits=probabilities(candidate,b);candidate=candidate.cpu();valid=b['attention_mask'].bool().cpu();x=reference[valid].float();y=candidate[valid].float();delta=y-x
    record={'attention_mode':mode,'ids':[r['id'] for r,_ in group],'hidden_max_abs_difference':float(delta.abs().max()),'hidden_relative_rms':float(delta.square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-12)),'logit_max_abs_difference':max(float((a-c).abs().max()) for a,c in zip(ref_logits,new_logits)),'calibrated_probability_max_abs_difference':max(float(((a/temperature).softmax(-1)-(c/temperature).softmax(-1)).abs().max()) for a,c in zip(ref_logits,new_logits)),'decisions':4,'argmax_agreements':sum(int(a.argmax()==c.argmax()) for a,c in zip(ref_logits,new_logits))}
    numerical.append(record);mode_numeric.append(record);write_json(out/'numerical.json',numerical);checkpoint();del reference,candidate,x,y,delta,b
   if handle is not None:
    handle.remove();delattr(text,'_research_attention_hook')
    for layer in text.layers:
     if layer.block_type=='full_attention':layer.self_attn.is_causal=True
   print(json.dumps({'event':'kernel_mode_complete','attention_mode':mode,'agreements':sum(r['argmax_agreements'] for r in mode_numeric),'decisions':32}),flush=True)
  unchanged=all(not p.requires_grad and p.grad is None and p._version==version for p,version in frozen)
  if not unchanged:raise RuntimeError('Kernel validation changed frozen parameters')
  numerical_ok=all(r['hidden_relative_rms']<=.02 and r['calibrated_probability_max_abs_difference']<=.02 and r['argmax_agreements']==r['decisions'] for r in numerical)
  batch_speed=[r['speedup'] for r in speed if r['batch_size']==4];speed_ok=min(batch_speed)>=1.2
  result={'status':'complete','adoptable':numerical_ok and speed_ok,'numerical_gate_passed':numerical_ok,'speed_gate_passed':speed_ok,'weights_unchanged':unchanged,'batch4_speedups':batch_speed,'median_all_workload_speedup':statistics.median(r['speedup'] for r in speed),'maximum_hidden_relative_rms':max(r['hidden_relative_rms'] for r in numerical),'maximum_probability_difference':max(r['calibrated_probability_max_abs_difference'] for r in numerical),'argmax_agreements':sum(r['argmax_agreements'] for r in numerical),'decisions':sum(r['decisions'] for r in numerical),'speed':speed,'numerical':numerical,'interpretation':'Warm backbone forwards only; model load, CPU cache hashing, JIT cold compilation and cached-head optimizer updates are separate. This dev screen is not proof of bitwise equality or full-benchmark equivalence.'}
  write_json(out/'summary.json',result);return {k:v for k,v in result.items() if k not in ['speed','numerical']}
 except Exception as exc:write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:
  for n,fn in optimized.items():setattr(qwen,n,fn)
  write_json(out/'checksums.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'});checkpoint()
