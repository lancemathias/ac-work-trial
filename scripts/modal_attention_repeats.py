"""Prepared attention replications. Launch only after releasing the old GPU worker."""
import json,subprocess,time,sys
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_attention_bidir import ROOT,ENV,DEADLINE,image as base_image,data,models,results,secret
app=modal.App('ac-worktrial-attention-repeats')
image=base_image
if modal.is_local():image=base_image.add_local_file(ROOT/'outputs/autoresearch/attention-repeat-preparation.json','/project/repeat-preparation.json')

def validated():
 from attention_repeat_protocol import validate_configs
 from training_data import digest,load_splits
 receipt=json.loads(Path('/project/repeat-preparation.json').read_text())
 configs=validate_configs('/project',receipt,json.loads(Path('/project/configs/attention-bidir-qwen38-27b-v3.json').read_text()))
 manifest=Path(receipt['fixed_manifest_remote'])
 if digest(manifest)!=receipt['fixed_manifest_sha256']:raise ValueError('Original split manifest changed')
 splits,m=load_splits(manifest)
 for role,sha in receipt['fixed_point_sha256'].items():
  if m['splits'][role]['points']['sha256']!=sha or len(splits[role])!=receipt['fixed_counts'][role]:raise ValueError('Original partition changed')
 return receipt,configs,manifest

@app.function(image=image,cpu=4,memory=16384,timeout=600,retries=0,volumes={'/results':results.read_only()})
def cpu_check():
 validated()
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_attention_repeat_protocol','test_checkpoint_selection','test_resident_backbone','test_cached_training'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-12000:])
 return {'status':'passed','tests':p.stderr,'fixed_split_manifest_validated':True}

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=163840,timeout=4*3600,retries=0,max_containers=1,scaledown_window=2,
 volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def repeat_all(commit,run_id):
 if time.time()+4*3600+60>DEADLINE:raise TimeoutError('Replication worst-case timeout exceeds reservation')
 import gc,torch
 from kernel_runtime import configure_optimized_runtime
 from resident_backbone import BackbonePool
 from attention_repeat_protocol import movement_probe
 from training_data import digest,validate_run_id
 from train_joint import run,write_json
 from evaluate_trained_joint import evaluate
 from report_trained_joint import report
 validate_run_id(run_id);receipt,configs,manifest=validated();configure_optimized_runtime('both')
 folder=Path('/results')/(run_id+'-pipeline');folder.mkdir(exist_ok=False);completed=[];movement=None;pool=BackbonePool()
 def status(phase,**kwargs):write_json(folder/'status.json',dict(phase=phase,commit=commit,completed=completed,**kwargs));results.commit()
 # Alternate pair order; every seed receives both modes, never best-of-seed selection.
 order=[(20261015,'native_causal'),(20261015,'bidirectional_full_attention'),(20261016,'bidirectional_full_attention'),(20261016,'native_causal')]
 try:
  with pool.activate():
   for seed,mode in order:
    cfg=next(c for c in configs if c['seed']==seed and c['attention_mode']==mode);name=cfg['run_name']
    status('training',current=name)
    if pool.model is not None and any(p.device.type!='cpu' for p in pool.model.parameters()):raise MemoryError('Park27B on CPU before allocating feature cache')
    cache=Path(receipt['caches'][mode]['directory']);cache_manifest=cache/'manifest.json'
    if digest(cache_manifest)!=receipt['caches'][mode]['manifest_sha256']:raise ValueError('Pinned feature cache changed')
    identity=json.loads(cache_manifest.read_text())['identity']
    if identity['attention_mode']!=mode:raise ValueError('Wrong attention cache')
    training=Path('/results')/name;test=Path('/results')/(name+'-test');cfgpath=Path('/tmp')/(name+'.json');write_json(cfgpath,cfg)
    run(cfgpath,manifest,training,checkpoint_callback=results.commit,feature_cache=dict(directory=str(cache),expected_manifest_sha256=digest(cache_manifest),expected_adapter_sha256=digest('/project/scripts/benchmark_training_data.py'),expected_source_config=identity['training_source_config'],expected_extraction_batch_size=4))
    summary=json.loads((training/'summary.json').read_text())
    if summary['steps']!=490:raise ValueError('Wrong update count')
    gc.collect();torch.cuda.empty_cache()
    if pool.model is not None:pool.model.to('cuda')
    status('benchmark',current=name);evaluate(training,test,commit,results.commit);metrics=report(test,label=name)
    if movement is None:
     movement=movement_probe(pool.model);write_json(folder/'offload-restore-proof.json',movement)
    pool.model.to('cpu');gc.collect();torch.cuda.empty_cache()
    write_json(test/'pipeline.json',{'status':'complete','training_run':name,'commit':commit})
    write_json(test/'checksums.json',{p.name:digest(p) for p in test.iterdir() if p.is_file() and p.name!='checksums.json'})
    completed.append({'seed':seed,'mode':mode,'run_id':name,'composite':metrics['macro_coverage_adjusted_accuracy'],'initial_head_sha256':digest(training/'initial-head.safetensors')})
    pair=[r for r in completed if r['seed']==seed]
    if len(pair)==2:
     if pair[0]['initial_head_sha256']!=pair[1]['initial_head_sha256']:raise ValueError('Seed pair initialization differs')
     traces=[[json.loads(l) for l in (Path('/results')/r['run_id']/'metrics.jsonl').read_text().splitlines() if json.loads(l)['event']=='train_step'] for r in pair]
     if len(traces[0])!=490 or [(r['step'],r['record_ids']) for r in traces[0]]!=[(r['step'],r['record_ids']) for r in traces[1]]:raise ValueError('Seed pair update order differs')
    status('condition_complete',model_loads=pool.loads,model_hits=pool.hits)
  status('complete',model_loads=pool.loads,model_hits=pool.hits);return {'status':'complete','completed':completed}
 except Exception as exc:status('failed',error=type(exc).__name__,message=str(exc));raise
 finally:pool.close();results.commit();models.commit()

@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=cpu_check.remote();(ROOT/'outputs/autoresearch/attention-repeat-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

# GPU entrypoint deliberately omitted until the current warm queue is complete
# and the existing GPU service/heartbeat have been safely released.
