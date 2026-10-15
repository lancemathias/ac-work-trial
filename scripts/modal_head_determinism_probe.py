"""Actual cached-feature backward repeat test; no data or head-recipe tuning."""
import json,sys,time,subprocess
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_warm_worker import ROOT,ENV,image,data,models,results,DEADLINE
app=modal.App('ac-worktrial-head-determinism-probe')
REFERENCE='ab1-fast-accuracy-seed20261014-existing-source-control-repeat'
CACHE='/data/hidden-cache/ab1-fast-accuracy-seed20261014-existing-source-control'

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=1800,retries=0,max_containers=1,scaledown_window=2,
 volumes={'/data':data.read_only(),'/models':models.read_only(),'/results':results})
def probe(commit,run_id):
 if time.time()+1860>DEADLINE:raise TimeoutError('Probe exceeds reservation')
 import gc,hashlib,statistics,torch,uuid
 from contextlib import nullcontext
 from torch.nn.attention import sdpa_kernel,SDPBackend
 from transformers import AutoProcessor
 from training_data import load_splits,digest,validate_run_id
 from train_joint import prepare,train_head,write_json
 from frozen_feature_cache import FrozenFeatureBackbone
 from vendor.clef_joint_schema import JointSchemaHead
 from kernel_runtime import configure_optimized_runtime
 from head_probe_io import prepare_run,seal
 validate_run_id(run_id);root=Path('/results')/run_id;root.mkdir(exist_ok=True)
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 runtime=configure_optimized_runtime('both')
 cfg=json.loads(Path('/project/configs/source-addbacks/existing-source-control.json').read_text())
 splits,manifest=load_splits('/project/source-addbacks/existing-source-control/manifest.json')
 identity=json.loads((Path(CACHE)/'manifest.json').read_text())['identity']
 baseline=Path('/results')/REFERENCE
 expected=[json.loads(l) for l in (baseline/'metrics.jsonl').read_text().splitlines() if json.loads(l)['event']=='train_step']
 expected_initial=digest(baseline/'initial-head.safetensors')
 request={'commit':commit,'run_id':run_id,'seed':cfg['seed'],'steps_per_replica':32,'replicas_per_backend':2,'source_config':cfg,'cache_manifest_sha256':digest(Path(CACHE)/'manifest.json'),'initial_head_sha256':expected_initial}
 complete=prepare_run(root,request);results.commit()
 if complete is not None:return complete
 attempts=root/'attempts';attempts.mkdir(exist_ok=True)
 if len(list(attempts.iterdir()))>=3:
  result={'status':'preemption_budget_exhausted','request':request,'scope':'No more than3 attempts; inspect external preemption before a fresh request.'};write_json(root/'result.json',result);seal(root);results.commit();return result
 attempt_id=uuid.uuid4().hex;folder=attempts/attempt_id;folder.mkdir(exist_ok=False)
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
 write_json(folder/'started.json',{'request':request,'attempt_id':attempt_id,'gpu':gpu,'time':time.time()});results.commit()
 tokenizer=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 prepared={role:prepare(rows,tokenizer,cfg['max_input_tokens']) for role,rows in splits.items()}
 backbone=FrozenFeatureBackbone(CACHE,cfg,prepared,torch.device('cuda'),digest(Path(CACHE)/'manifest.json'),digest('/project/scripts/benchmark_training_data.py'),expected_source_config=identity['training_source_config'],expected_extraction_batch_size=4)
 class ProbeComplete(Exception):pass
 def tensor_hash(named):
  h=hashlib.sha256()
  for name,tensor in named:
   if tensor is not None:h.update(name.encode()+b'\0');h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 def once(mode,number):
  torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
  head=JointSchemaHead(hidden_size=backbone.get_output_embeddings().weight.shape[1],**cfg['head']).float().cuda()
  out=Path('/tmp')/(attempt_id+'-'+mode+'-'+str(number));out.mkdir(exist_ok=False);rows=[];operations=set()
  def hook(module,args,outputs):
   if operations or not torch.is_grad_enabled():return
   stack=[v.grad_fn for group in outputs for v in group];seen=set()
   while stack:
    node=stack.pop()
    if node is None or node in seen:continue
    seen.add(node);name=type(node).__name__
    if any(x in name for x in ['Attention','Softmax','Bmm']):operations.add(name)
    stack.extend(n for n,_ in node.next_functions)
  handle=head.register_forward_hook(hook)
  def emit(event):
   if event['event']!='train_step':return
   ref=expected[event['step']-1]
   if event['record_ids']!=ref['record_ids'] or event['learning_rate']!=ref['learning_rate']:raise ValueError('Probe differs from actual production windows/schedule')
   rows.append({k:event[k] for k in ['step','loss','gradient_norm','decisions_per_second']})
   rows[-1].update(parameters_sha256=tensor_hash(head.state_dict().items()),gradients_sha256=tensor_hash((n,p.grad) for n,p in head.named_parameters()),window_ids_sha256=hashlib.sha256(json.dumps(event['record_ids']).encode()).hexdigest())
   if event['step']==32:raise ProbeComplete()
  context=sdpa_kernel([SDPBackend.MATH]) if mode=='math' else nullcontext()
  began=time.monotonic()
  try:
   with context:
    flags={'flash':torch.backends.cuda.flash_sdp_enabled(),'efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}
    try:train_head(backbone,head,prepared,tokenizer.pad_token_id,torch.device('cuda'),cfg,out,emit)
    except ProbeComplete:pass
   initial=digest(out/'initial-head.safetensors')
   if initial!=expected_initial or len(rows)!=32:raise ValueError('Incorrect production initialization or step count')
   return {'initial_head_sha256':initial,'steps':rows,'attention_backward_operations':sorted(operations),'backend_flags':flags,'elapsed_seconds':time.monotonic()-began,'median_decisions_per_second':statistics.median(r['decisions_per_second'] for r in rows[2:])}
  finally:handle.remove();del head;gc.collect();torch.cuda.empty_cache()
 trials={};comparisons={}
 try:
  for mode in ['auto','math']:
   trials[mode]=[once(mode,n) for n in [1,2]];a,b=trials[mode]
   equal=lambda key:all(x[key]==y[key] for x,y in zip(a['steps'],b['steps']))
   comparisons[mode]={'parameters_bitwise_equal':equal('parameters_sha256'),'gradients_bitwise_equal':equal('gradients_sha256'),'losses_exact':equal('loss'),'windows_equal':equal('window_ids_sha256'),'maximum_loss_difference':max(abs(x['loss']-y['loss']) for x,y in zip(a['steps'],b['steps']))}
   write_json(folder/'progress.json',{'completed_mode':mode,'comparisons':comparisons});results.commit()
  backbone.verify_unchanged();accepted=all(comparisons['math'][k] for k in ['parameters_bitwise_equal','gradients_bitwise_equal','losses_exact','windows_equal'])
  result={'status':'math_fixture_gate_passed' if accepted else 'math_fixture_gate_failed','request':request,'attempt_id':attempt_id,'gpu':gpu,'commit':commit,'torch':str(torch.__version__),'runtime':runtime,'seed':cfg['seed'],'production_recipe':cfg,'cache_manifest_sha256':digest(Path(CACHE)/'manifest.json'),'initial_head_sha256':expected_initial,'comparisons':comparisons,'trials':trials,'scope':'Two32-step replicas per backend on actual production initial head/features/windows/LR schedule. No data, architecture or optimizer tuning. This is not full-run reproducibility or a model-quality comparison; full matched controls remain required. All pairs belong to this attempt/GPU; partial attempts never supply comparison rows.'}
  write_json(folder/'result.json',result);write_json(root/'result.json',result);seal(root);results.commit();return result
 finally:results.commit()

@app.local_entrypoint()
def main(run_id:str='head-determinism-probe-v1'):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 hold=json.loads((ROOT/'outputs/autoresearch/source-reproducibility-hold.json').read_text())
 if hold['phase']!='held_after_complete_control':raise RuntimeError('Wait for completed-control safe hold; never allocate a second GPU')
 current=json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal','app','list','--profile','ac-worktrial','--env',ENV,'--json'],text=True))
 if any(a['app_id'] in ['ap-3tDGWsFm6ZdCIh4XjJOeF7','ap-iPVnuNNE1KPK5wCkbvabNw'] and a['state']!='stopped' for a in current):raise RuntimeError('Old queue/service has not stopped')
 containers=json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal','container','list','--profile','ac-worktrial','--env',ENV,'--json'],text=True))
 if any(c['app_id'] in ['ap-3tDGWsFm6ZdCIh4XjJOeF7','ap-iPVnuNNE1KPK5wCkbvabNw'] for c in containers):raise RuntimeError('Wait for old containers to exit before allocating the probe GPU')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();result=probe.remote(commit,run_id)
 (ROOT/'outputs/autoresearch'/('head-determinism-probe-'+run_id+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'comparisons':result.get('comparisons')}))
