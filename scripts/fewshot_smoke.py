"""Label-blind real-model capacity probe; no fitting or accuracy computation."""
import hashlib,json,time
from pathlib import Path

def smoke(output,deadline,checkpoint=lambda:None):
 import torch
 from run_baselines import ROOT
 from training_data import load_splits,digest,canonical
 from head_probe_io import write_json,seal
 from resident_backbone import get_processor,get_model
 from kernel_runtime import configure_optimized_runtime
 from fewshot_pipeline import prepare,predict
 from fewshot_capacity import select
 from thinking_smoke_plan import reservation_gate
 out=Path(output)
 if out.exists():raise FileExistsError('Preserve prior capacity evidence')
 if time.time()+4260>=deadline:raise TimeoutError('Insufficient reservation for capacity smoke and idle/margin')
 proof_path=Path('/project/fewshot-quality-v2-preparation.json');proof=json.loads(proof_path.read_text());runtime_proof=json.loads(Path('/project/fewshot-quality-v2-runtime-cpu.json').read_text())
 if proof['status']!='passed' or runtime_proof['status']!='passed':raise ValueError('CPU preparation and runtime contracts required')
 for receipt in [proof,runtime_proof]:
  for name,expected in receipt['source_sha256'].items():
   if digest(ROOT/name)!=expected:raise ValueError('CPU-tested source changed: '+name)
 settings=json.loads((ROOT/'configs/fewshot-seven/native7-quality-v2.json').read_text());path=Path(settings['demonstrations_path'])
 if settings['batch_size']!=4 or settings['final_prompt_cap']!=12288:raise ValueError('Fixed batch4 and12288 context capacity plan required')
 if digest(path)!=settings['demonstrations_sha256'] or digest(proof_path)!=settings['preparation_sha256']:raise ValueError('Locked demonstrations or preparation changed')
 settings=dict(settings,_demonstrations=json.loads(path.read_text()));out.mkdir();checkpoint()
 try:
  runtime=configure_optimized_runtime('both');cfg=json.loads((ROOT/'configs/e4-qwen38-27b-v3.json').read_text())
  if any(settings[k]!=cfg[k] for k in ['backbone_id','backbone_revision']):raise ValueError('Backbone differs')
  splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json');tokenizer=get_processor(cfg).tokenizer;prepared=prepare(tokenizer,splits['dev'],settings,training_sources=True);h=hashlib.sha256()
  for row in prepared:h.update(json.dumps({'id':row['id'],**row['encoding']},sort_keys=True,separators=(',',':')).encode())
  expected=proof['sets']['dev']
  if len(prepared)!=expected['records'] or h.hexdigest()!=expected['encodings_sha256']:raise ValueError('Dev prompt fingerprints differ')
  chosen=select(prepared);write_json(out/'protocol.json',{'selection':'8 dev length quantiles plus4 synthetic12288-token prompts, metadata labels removed','selected_ids':[r['id'] for r in chosen],'kernel_runtime':runtime,'test_read':False,'accuracy_computed':False,'source_sha256':{n:digest(ROOT/n) for n in ['scripts/fewshot_smoke.py','scripts/fewshot_capacity.py']}})
  torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
  model=get_model(cfg);frozen=[(p,p._version) for p in model.parameters()];torch.cuda.reset_peak_memory_stats();times=[];emitted=[]
  for start in range(0,len(chosen),4):
   if time.time()+2460>=deadline:raise TimeoutError('Reservation guard before capacity batch')
   tick=time.monotonic();items=predict(model,tokenizer,chosen[start:start+4],settings,emit=lambda _:None);torch.cuda.synchronize();times.append(time.monotonic()-tick);emitted.extend(items);checkpoint()
  unchanged=all(not p.requires_grad and p.grad is None and p._version==v for p,v in frozen)
  if not unchanged:raise RuntimeError('Frozen parameters changed')
  (out/'smoke-logits.jsonl').write_text(''.join(canonical(r)+'\n' for r in emitted))
  capacity=sum(r['id'].startswith('synthetic-fewshot-capacity-') and r['input_tokens']==12288 for r in emitted)==4
  value={'status':'complete','capacity_validated':capacity,'capacity_gate':reservation_gate(times,time.time(),deadline,batch_size=4),'parameters_unchanged':unchanged,'records':len(emitted),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'optimizer_updates':0,'test_read':False,'accuracy_computed':False,'qualification':'Synthetic repeated public tokens check capacity and timing only; strict deadline enforced again during full inference.'};write_json(out/'result.json',value);return value
 except Exception as exc:write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:seal(out);checkpoint()
