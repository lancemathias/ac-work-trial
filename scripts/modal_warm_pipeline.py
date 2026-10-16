"""Hand off waiting experiments to one deployed resident GPU worker."""
import json,time,subprocess,threading
from pathlib import Path
import modal
import sys
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_warm_worker import ROOT,ENV,DEADLINE,IDLE_SECONDS,APP_NAME,image,data,results
BIDIR='e5-fast-accuracy-bidir-v3-seed20261014';BATCH='ab1-fast-accuracy-seed20261014'
PROBE='ab1-fast-accuracy-polarity-seed20261014';READOUT='readout-fast-qwen38-27b-v1'
app=modal.App('ac-worktrial-warm-research-driver')
@app.function(image=image,cpu=4,memory=16384,timeout=600,retries=0,scaledown_window=2)
def cpu_check():
 import sys
 p=subprocess.run([sys.executable,'/project/scripts/kernel_cpu_test_bootstrap.py','test_resident_backbone','test_cache_composition','test_binary_views','test_backbone_attention','test_pretrained_readout','test_pretrained_readout_pipeline','test_polarity_diagnostic','test_checkpoint_selection','test_training','test_cached_training','test_training_runtime','test_adamw_groups','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-18000:])
 return {'status':'passed','tests':p.stderr}
@app.function(image=image,cpu=8,memory=32768,timeout=24*3600,retries=0,max_containers=1,volumes={'/data':data,'/results':results})
def pipeline(commit,driver_id):
 import sys
 sys.path.insert(0,'/project/scripts')
 from modal_source_addbacks import NAMES,verify_cpu,compose_conditions
 from modal_polarity_diagnostic import verify_probe
 worker=modal.Cls.from_name(APP_NAME,'GPUWorker',environment_name=ENV)()
 root=Path('/results')/driver_id;root.mkdir(exist_ok=False);heartbeat_stop=threading.Event()
 def keep_warm():
  while not heartbeat_stop.wait(300):
   if time.time()+IDLE_SECONDS+60>=DEADLINE:return
   try:
    state=worker.ping.remote()
    if state['failed']:return
   except Exception:return  # Main job/driver owns failure diagnosis; never retry in a loop.
 def status(folder,phase,**kw):
  folder.mkdir(exist_ok=True);(folder/'status.json').write_text(json.dumps({'phase':phase,'commit':commit,'time':time.time(),**kw},indent=2)+'\n');results.commit()
 def take_over(run):
  folder=Path('/results')/(run+'-pipeline');folder.mkdir(exist_ok=True)
  if (folder/'status.json').exists():
   previous=json.loads((folder/'status.json').read_text())
   if previous['phase'] not in ['waiting_for_e5','waiting_for_source_grid','waiting_for_source_grid_and_polarity']:raise ValueError('Cannot take over an active or completed GPU stage: '+run)
   (folder/'warm-takeover.json').write_text(json.dumps({'previous_status':previous,'new_driver':driver_id,'commit':commit},indent=2)+'\n')
  return folder
 try:
  status(root,'cpu_validation');checks=cpu_check.remote();(root/'cpu-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
  source_check=verify_cpu.local();probe_check=verify_probe.local()
  if source_check['status']!='passed' or probe_check['status']!='passed':raise RuntimeError('Data/recipe validation failed')
  (root/'data-validation.json').write_text(json.dumps({'sources':source_check,'probe':probe_check},indent=2)+'\n');results.commit()
  status(root,'waiting_for_e5')
  while True:
   if time.time()+7200+IDLE_SECONDS+60>DEADLINE:raise TimeoutError('Insufficient reservation for remaining grid')
   results.reload();state=Path('/results')/(BIDIR+'-pipeline')/'status.json'
   if state.exists():
    phase=json.loads(state.read_text())['phase']
    if phase=='failed':raise RuntimeError('BIDIR failed; no new GPU allocated')
    if phase=='complete':break
   time.sleep(20)
  # Two separate RPCs prove actual container/model reuse and identical output.
  status(root,'resident_reuse_validation')
  first=worker.run_job.remote('smoke','warm-smoke-cold-v1',{})
  second=worker.run_job.remote('smoke','warm-smoke-reuse-v1',{})
  if first['worker_id']!=second['worker_id'] or second['model_loads_after']!=1 or first['result']['hidden_sha256']!=second['result']['hidden_sha256']:raise RuntimeError('Resident reuse proof failed')
  (root/'resident-reuse-proof.json').write_text(json.dumps({'cold':first,'warm':second},indent=2)+'\n');results.commit()
  threading.Thread(target=keep_warm,daemon=True).start()
  sources=take_over(BATCH);status(sources,'extracting_union',worker=first['worker_id'])
  worker.run_job.remote('extract_sources','warm-source-setup-v1',{})
  status(sources,'composing_condition_caches');caches=compose_conditions.local(BATCH)
  (sources/'cache-receipts.json').write_text(json.dumps(caches,indent=2)+'\n');results.commit();completed=[]
  for name in NAMES:
   status(sources,'training_and_benchmark',current=name,completed=completed)
   result=worker.run_job.remote('source_condition','warm-'+name+'-v1',{'name':name,'run_id':BATCH+'-'+name,'commit':commit,'cache':caches[name]})['result']
   if completed and result['initial_head_sha256']!=completed[0]['initial_head_sha256']:raise ValueError('Matched source heads have different initial weights')
   completed.append(result);status(sources,'condition_complete',completed=completed)
  status(sources,'complete',completed=completed)
  probe=take_over(PROBE);status(probe,'evaluating_locked_heads')
  diagnostic=worker.run_job.remote('polarity','warm-polarity-v1',{'probe_id':PROBE,'commit':commit})['result'];status(probe,'complete',result=diagnostic)
  readout=take_over(READOUT);status(readout,'frozen_readout')
  candidate=worker.run_job.remote('readout','warm-readout-v1',{'run_id':READOUT,'commit':commit})['result'];status(readout,'complete',result=candidate)
  status(root,'complete_keep_warm',completed=completed,readout=candidate)
  while time.time()+IDLE_SECONDS+60<DEADLINE:
   time.sleep(min(300,max(0,DEADLINE-IDLE_SECONDS-60-time.time())))
  status(root,'complete_reservation_expiry');return {'status':'complete','completed':completed,'readout':candidate}
 except Exception as exc:status(root,'failed',error=type(exc).__name__,message=str(exc));raise
 finally:heartbeat_stop.set()
@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=cpu_check.remote();(ROOT/'outputs/autoresearch/warm-worker-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

@app.function(image=image,cpu=4,memory=16384,timeout=600,retries=0,scaledown_window=2,
 volumes={'/models':modal.Volume.from_name('ac-worktrial-models',environment_name=ENV).read_only(),'/data':data.read_only()})
def check_memory():
 """Conservative resident-model/cache budget using actual pinned token lengths."""
 import sys
 sys.path.insert(0,'/project/scripts')
 from transformers import AutoProcessor
 from jsonschema import Draft202012Validator
 from benchmark_training_data import adapt_training_point
 from vendor.clef_joint_schema import encode_record
 from training_data import digest
 from modal_source_addbacks import NAMES
 cfg=json.loads(Path('/project/configs/e3-medium-v3.json').read_text());revision=cfg['backbone_revision']
 base=Path('/data/hidden-cache/qwen35-9b-v3-batch4-ef15c46f4919');manifest=json.loads((base/'manifest.json').read_text())
 dim=manifest['identity']['hidden_size'];index=json.loads((base/'index.json').read_text());files={f['path']:f for f in manifest['files']}
 base_bytes=sum(files[n]['bytes'] for n in {r['shard'] for r in index});lexical_bytes=files['output-embedding-rows.safetensors']['bytes']
 snapshot=Path('/models/hub/models--Qwen--Qwen3.5-9B/snapshots')/revision
 # 5% allowance above checkpoint tensor bytes, plus 8GiB for head/workspace.
 model_bytes=int(json.loads((snapshot/'model.safetensors.index.json').read_text())['metadata']['total_size']*1.05)
 tokenizer=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=revision,cache_dir='/models/hub',local_files_only=True).tokenizer
 union=Path('/project/source-addbacks/union-additions.jsonl');source=json.loads(Path('/project/configs/source-addbacks/union-cache-source.json').read_text())
 if digest(union)!=source['config']['config']['dataset']['sha256']:raise ValueError('Union data changed')
 validator=Draft202012Validator(json.loads(Path('/project/configs/normalized-training.schema.json').read_text()));encoded=[];lexical=set()
 for line in union.open():
  row=adapt_training_point(json.loads(line),validator);e=encode_record(tokenizer,dict(row['request'],id=row['id']),max_length=10**9)
  if len(e.input_ids)>8192:raise ValueError('Overlength add-back input; no GPU trial')
  encoded.append((row['id'],e))
  for q in e.questions:
   for start,end in q.option_spans:lexical.update(e.input_ids[start:end])
 encoded.sort(key=lambda x:(len(x[1].input_ids),x[0]));blocks={};lookup={}
 for begin in range(0,len(encoded),64):
  group=encoded[begin:begin+64];blocks[begin]=sum(len(e.input_ids)*(dim*2+8) for _,e in group)+4096
  for rid,_ in group:lookup[rid]=begin
 conditions={};budget=80_000_000_000;workspace=8*1024**3
 for name in NAMES:
  ids={json.loads(x)['id'] for x in Path('/project/source-addbacks',name,'additions.jsonl').open()}
  shard_bytes=sum(blocks[b] for b in {lookup[rid] for rid in ids})
  estimate=model_bytes+base_bytes+shard_bytes+lexical_bytes+len(lexical)*dim*2+workspace
  conditions[name]={'estimated_bytes_with_allowances':estimate,'headroom_bytes_vs_conservative80GB':budget-estimate}
  if estimate>=budget:raise MemoryError('Resident cache needs explicit model offload: '+name)
 return {'status':'passed','backbone_id':cfg['backbone_id'],'hidden_size':dim,'union_records':len(encoded),'maximum_union_tokens':max(len(e.input_ids) for _,e in encoded),'model_bytes_with5percent_allowance':model_bytes,'workspace_allowance_bytes':workspace,'gpu_budget_bytes':budget,'conditions':conditions,'scope':'Conservative CPU estimate from actual token lengths and used64-record shards; actual GPU peak remains to be measured.'}

@app.local_entrypoint()
def memory():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=check_memory.remote();(ROOT/'outputs/autoresearch/warm-worker-memory-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
@app.local_entrypoint()
def main(driver_id:str="warm-research-pipeline-v2"):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();print(json.dumps(pipeline.remote(commit,driver_id),indent=2))
