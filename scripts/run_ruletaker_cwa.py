"""Future same-record RuleTaker repair, using one activated resident9B pool."""
import json,time,uuid,gc
from pathlib import Path

def run(output,commit,worker,checkpoint,data_checkpoint,deadline):
 import torch
 from run_baselines import ROOT
 from training_data import digest,load_splits
 from head_probe_io import prepare_run,write_json,seal
 from cache_hidden_runtime_batch4 import extract
 from compose_feature_cache import compose
 from math_source_repro import runtime_config
 from train_joint import run as train
 from evaluate_trained_joint import evaluate
 from report_trained_joint import report
 from report_prediction_dominance import summarize
 root=Path(output);cfgpath=ROOT/'configs/source-quality/ruletaker-cwa-explicit.json';cfg=json.loads(cfgpath.read_text());dataset=Path('/project/ruletaker-cwa-explicit')
 proof=json.loads(Path('/project/ruletaker-cwa-cpu.json').read_text())
 if proof['status']!='passed' or digest(cfgpath)!=proof['source_sha256']['configs/source-quality/ruletaker-cwa-explicit.json']:raise ValueError('CPU-tested CWA training recipe changed')
 if digest(dataset/'dataset.jsonl')!=proof['dataset_sha256']:raise ValueError('CPU-tested CWA records changed')
 if time.time()+3600+1200+60>=deadline:raise TimeoutError('Insufficient reservation including warm idle expiry')
 request={'kind':'same250_explicit_closed_world','commit':commit,'runner_sha256':digest(__file__),'config_sha256':digest(cfgpath),'dataset_sha256':digest(dataset/'dataset.jsonl'),'initial_head_sha256':'909b3c9ff1f565479d2bd81a1346564af2f5f0a433f2bf869fa9e7cad4ed573e'}
 complete=prepare_run(root,request);checkpoint()
 if complete is not None:return complete
 attempts=root/'attempts';attempts.mkdir(exist_ok=True)
 if len(list(attempts.iterdir()))>=3:raise RuntimeError('Three interrupted attempts; diagnose before any new submission')
 attempt=attempts/uuid.uuid4().hex;attempt.mkdir();write_json(attempt/'started.json',{'request':request,'worker':worker,'time':time.time()});checkpoint()
 try:
  snapshot=json.loads((ROOT/'configs/source-quality/ruletaker-cwa-cache-source.json').read_text())
  additions=Path('/data/hidden-cache/ruletaker-cwa250-fast-v1');base=Path('/data/hidden-cache/qwen35-9b-v3-fast-v1');combined=Path('/data/hidden-cache/ruletaker-cwa-explicit-fast-v1')
  extract(snapshot,dataset/'additions.jsonl',ROOT/'configs/normalized-training.schema.json',additions,checkpoint=data_checkpoint)
  splits,_=load_splits(dataset/'manifest.json');rows=[r for values in splits.values() for r in values];parts=[{'directory':str(p),'manifest_sha256':digest(p/'manifest.json')} for p in [base,additions]]
  if combined.exists():
   # A partial composition cannot be treated as complete or silently overwritten.
   manifest=json.loads((combined/'manifest.json').read_text())
   if manifest['status']!='complete' or manifest['identity']['parent_manifest_sha256']!=[p['manifest_sha256'] for p in parts] or manifest['identity']['training_source_config']!=cfg:raise ValueError('Existing composed cache differs or is incomplete')
  else:manifest=compose(parts,combined,rows,cfg,allow_source_subset=True);data_checkpoint()
  if manifest['identity']['record_count']!=8178:raise ValueError('Expected7928base plus250 changed requests')
  runtime=attempt/'runtime-config.json';write_json(runtime,runtime_config(cfg,root.name));write_json(attempt/'cache-receipt.json',{'directory':str(combined),'manifest_sha256':digest(combined/'manifest.json'),'parents':parts})
  gc.collect();torch.cuda.empty_cache()
  training=attempt/'training';test=attempt/'test'
  train(runtime,dataset/'manifest.json',training,checkpoint_callback=checkpoint,feature_cache={'directory':str(combined),'expected_manifest_sha256':digest(combined/'manifest.json'),'expected_adapter_sha256':digest(ROOT/'scripts/benchmark_training_data.py'),'expected_source_config':cfg,'expected_extraction_batch_size':4},expected_initial_head_sha256=request['initial_head_sha256'])
  summary=json.loads((training/'summary.json').read_text())
  if summary['steps']!=510:raise ValueError('Fixed510-update recipe required')
  gc.collect();torch.cuda.empty_cache();evaluate(training,test,commit,checkpoint)
  metrics=report(test,label='Same-record explicit closed-world RuleTaker');write_json(test/'prediction-distributions.json',summarize(test/'calibrated-predictions.jsonl'))
  import wandb
  try:
   wb=json.loads((training/'wandb-run.json').read_text());remote=wandb.Api().run('lmathias/ac-worktrial-decisions/'+wb['id']);remote.summary['test_composite']=metrics['macro_coverage_adjusted_accuracy'];remote.summary['test_per_suite']=metrics['suites'];remote.summary.update()
  except Exception as exc:write_json(test/'wandb-update-warning.json',{'error':type(exc).__name__})
  seal(test)
  value={'status':'complete','request':request,'worker':worker,'attempt':str(attempt.relative_to(root)),'composite':metrics['macro_coverage_adjusted_accuracy'],'steps':510}
  write_json(root/'result.json',value);seal(root);checkpoint();return value
 finally:checkpoint();data_checkpoint()
