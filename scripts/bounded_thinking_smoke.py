"""Run before a full thinking trial, inside the future resident 27B worker."""
import hashlib,json,time
from pathlib import Path

def smoke(output,deadline,checkpoint=lambda:None):
 import torch
 from run_baselines import ROOT
 from training_data import digest,load_splits,canonical
 from head_probe_io import write_json,seal
 from resident_backbone import get_processor,get_model
 from kernel_runtime import configure_optimized_runtime
 from bounded_thinking_readout import prepare,predict
 from thinking_smoke_plan import select,reservation_gate
 out=Path(output)
 if out.exists():raise FileExistsError('Preserve prior smoke evidence; diagnose before another run')
 proof_path=Path('/project/bounded-thinking-cpu.json')
 proof=json.loads(proof_path.read_text())
 if proof['status']!='passed':raise ValueError('Passed generation/cache CPU contracts required')
 for name,expected in proof['source_sha256'].items():
  if digest(ROOT/name)!=expected:raise ValueError('CPU-tested source changed: '+name)
 settings=json.loads((ROOT/'configs/bounded-thinking128.json').read_text())
 if time.time()+2400+1200+600>=deadline:raise TimeoutError('Insufficient reservation for smoke, idle expiry and margin')
 out.mkdir();checkpoint()
 try:
  runtime=configure_optimized_runtime('both')
  cfg=json.loads((ROOT/'configs/e4-qwen38-27b-v3.json').read_text())
  if any(settings[k]!=cfg[k] for k in ['backbone_id','backbone_revision']):raise ValueError('Smoke backbone differs')
  splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json')
  tokenizer=get_processor(cfg).tokenizer
  prepared=prepare(tokenizer,splits['dev'],settings,training_sources=True)
  h=hashlib.sha256()
  for row in prepared:h.update(json.dumps({'id':row['id'],**row['encoding']},sort_keys=True,separators=(',',':')).encode())
  expected=proof['sets']['dev']
  if len(prepared)!=expected['records'] or h.hexdigest()!=expected['encodings_sha256']:raise ValueError('Smoke dev inputs differ from CPU proof')
  chosen=select(prepared)
  # Synthetic capacity probe: same public dev tokens, extended without labels.
  # Exercises four maximum-length prompts through generation and final prefill.
  import copy
  longest=max(chosen,key=lambda x:len(x['encoding']['input_ids']))
  for i in range(settings['batch_size']):
   row=copy.deepcopy(longest);row['id']='synthetic-capacity-'+str(i)
   ids=row['encoding']['input_ids'];row['encoding']['input_ids']=(ids*((settings['thinking_prompt_cap']+len(ids)-1)//len(ids)))[:settings['thinking_prompt_cap']]
   chosen.append(row)
  write_json(out/'protocol.json',{'settings':settings,'kernel_runtime':runtime,'cpu_proof_sha256':digest(proof_path),'selection':'8 dev length quantiles plus4 synthetic maximum-context capacity probes; labels discarded','selected_ids':[x['id'] for x in chosen],'test_read':False,'runner_sha256':digest(__file__)})
  torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
  model=get_model(cfg);batch_times=[];emitted=[]
  # Explicit batch calls include synchronization and allocation.
  for begin in range(0,len(chosen),settings['batch_size']):
   if time.time()+600+1200+600>=deadline:raise TimeoutError('Reservation guard before smoke batch')
   tick=time.monotonic();items=predict(model,tokenizer,chosen[begin:begin+settings['batch_size']],settings)
   torch.cuda.synchronize();batch_times.append(time.monotonic()-tick)
   for item in items:
    with (out/'smoke-logits.jsonl').open('a') as stream:stream.write(canonical(item)+'\n')
   emitted.extend(items);checkpoint()
  gate=reservation_gate(batch_times,time.time(),deadline,batch_size=settings['batch_size'])
  value={'status':'complete','capacity_gate':gate,'records':len(emitted),'parameters_unchanged':True,'optimizer_updates':0,'test_read':False,'accuracy_computed':False,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'capacity_validated':any(x['id'].startswith('synthetic-capacity-') and x['expanded_tokens']>=settings['thinking_prompt_cap'] for x in emitted),'qualification':'Synthetic maximum-context prompts test memory and timing, not semantic quality. Capacity estimate is conservative but full runner still enforces deadline each batch.'}
  write_json(out/'result.json',value);return value
 except Exception as exc:
  write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:seal(out);checkpoint()
