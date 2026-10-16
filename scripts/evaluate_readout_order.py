"""Preregistered native/reverse probability pool, with test behind a dev gate."""
import hashlib,json,math,time,uuid,shutil
from pathlib import Path

def evaluate(output,commit,checkpoint=lambda:None):
 from run_baselines import ROOT
 from training_data import digest
 from head_probe_io import prepare_run,write_json,seal
 root=Path(output);baseline=Path('/results/readout-fast-qwen38-27b-v1');preflight_path=Path('/project/readout-order-preflight.json')
 preflight=json.loads(preflight_path.read_text())
 if preflight['status']!='passed':raise ValueError('CPU preparation required')
 for name,expected in preflight['source_sha256'].items():
  if digest(ROOT/name)!=expected:raise ValueError('Prepared source changed: '+name)
 request={'kind':'native_reverse_equal_probability_pool','commit':commit,'baseline_checksums_sha256':digest(baseline/'checksums.json'),'preflight_sha256':digest(preflight_path),'runner_sha256':digest(__file__),'pool_sha256':digest(ROOT/'scripts/readout_order_pool.py')}
 prior=prepare_run(root,request);checkpoint()
 if prior is not None:return prior
 attempts=root/'attempts';attempts.mkdir(exist_ok=True)
 if len(list(attempts.glob('*-started.json')))>=3:raise RuntimeError('Three interrupted attempts; diagnose before another run')
 attempt_id=uuid.uuid4().hex;attempt=attempts/attempt_id
 write_json(attempts/(attempt_id+'-started.json'),dict(request=request,time=time.time()));checkpoint()
 try:
  value=_evaluate_once(attempt,commit,checkpoint,baseline,preflight,root.name)
  for p in attempt.iterdir():
   if p.is_file() and p.name!='checksums.json':shutil.copy2(p,root/p.name)
  value=dict(value,request=request,attempt_id=attempt_id,run_id=root.name)
  write_json(root/'result.json',value);seal(root);checkpoint();return value
 finally:checkpoint()

def _evaluate_once(out,commit,checkpoint,baseline,preflight,run_name):
 import torch,wandb
 from train_joint import fit_temperature,metrics,slice_metrics
 from training_data import load_splits,digest,canonical
 from head_probe_io import write_json,seal
 from run_baselines import ROOT,load_panel,record,score
 from pretrained_readout import predict,SYSTEM_PROMPT
 from evaluate_pretrained_readout import prepare_decisions
 from evaluate_trained_joint import answer_body
 from readout_order_pool import pool
 from resident_backbone import get_processor,get_model
 from kernel_runtime import configure_optimized_runtime
 out=Path(out);out.mkdir(exist_ok=False);baseline=Path(baseline);started=time.monotonic();wb=None
 checks=json.loads((baseline/'checksums.json').read_text())
 def verified_json(name):
  path=baseline/name
  if digest(path)!=checks[name]:raise ValueError('Baseline artifact changed: '+name)
  return json.loads(path.read_text())
 def original(role):
  name=role+'-logits.jsonl';path=baseline/name
  if digest(path)!=checks[name]:raise ValueError('Baseline logits changed: '+role)
  return [json.loads(x) for x in path.read_text().splitlines()]
 prior=verified_json('completion.json');oldcfg=verified_json('config.json')
 if prior['status']!='complete' or not prior['parameters_unchanged']:raise ValueError('Completed immutable baseline required')
 cfg=json.loads((ROOT/'configs/e4-qwen38-27b-v3.json').read_text())
 if (cfg['backbone_id'],cfg['backbone_revision'])!=(oldcfg['backbone_id'],oldcfg['backbone_revision']):raise ValueError('Frozen backbone differs')
 runtime=configure_optimized_runtime('both')
 protocol={'kernel_runtime':runtime,'backbone_id':cfg['backbone_id'],'backbone_revision':cfg['backbone_revision'],'system_prompt':SYSTEM_PROMPT,'attention_mode':'native_causal','thinking':False,'option_orders':['native','reverse'],'pool':'equal arithmetic mean of uncalibrated semantic-option probabilities','temperature_role':'existing857-example nonbenchmark calibration only','selection':'pooled809-example dev accuracy gain>=1pp; reverse-only diagnostic never selected','parent_checkpoint_sha256':oldcfg['checkpoint_sha256'],'panel_sha256':oldcfg['panel_sha256'],'commit':commit,'runner_sha256':digest(__file__),'max_input_tokens':8192,'batch_size':4,'latency_definition':'Sum of amortized original and reverse forward latencies; excludes encoding, scoring and pooling'}
 protocol['checkpoint_sha256']=hashlib.sha256(canonical(protocol).encode()).hexdigest();write_json(out/'config.json',protocol)
 splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json')
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 try:
  wb=wandb.init(project='ac-worktrial-decisions',name=run_name,job_type='readout-order-pool',config=protocol);write_json(out/'wandb-run.json',{'id':wb.id,'url':wb.url});checkpoint()
  processor=get_processor(cfg);model=get_model(cfg);frozen=[(p,p._version) for p in model.parameters()]
  def infer(role,rows):
   prepared=prepare_decisions(processor.tokenizer,rows,training_sources=role!='test',option_order='reverse');h=hashlib.sha256()
   for x in prepared:h.update(json.dumps({'id':x['id'],**x['encoding']},sort_keys=True,separators=(',',':')).encode())
   spec=preflight['sets'][role]['reverse']
   if len(prepared)!=spec['records'] or h.hexdigest()!=spec['encodings_sha256']:raise ValueError('Reverse input differs from CPU preflight: '+role)
   emitted=0
   def emit(item):
    nonlocal emitted
    with (out/('reverse-'+role+'-logits.jsonl')).open('a') as f:f.write(canonical(item)+'\n')
    emitted+=1
    if emitted%100==0:checkpoint()
   value=predict(model,processor.tokenizer,prepared,batch_size=4,emit=emit)
   if len(value)!=len(prepared) or len({r['id'] for r in value})!=len(value):raise ValueError('Incomplete or duplicate reverse predictions')
   checkpoint();return value
  reverse_cal=infer('calibration',splits['calibration']);pooled_cal=pool(original('calibration'),reverse_cal)
  fitted=fit_temperature(pooled_cal);write_json(out/'temperature.json',fitted);protocol['temperature']=fitted['temperature'];write_json(out/'config.json',protocol)
  reverse_dev=infer('dev',splits['dev']);native_dev=original('dev');pooled_dev=pool(native_dev,reverse_dev)
  before=metrics(native_dev);after=metrics(pooled_dev);gate={'status':'passed' if after['accuracy']-before['accuracy']>=.01-1e-12 else 'failed','accuracy_before':before['accuracy'],'accuracy_after':after['accuracy'],'required_gain':.01,'observed_gain':after['accuracy']-before['accuracy']}
  development={'native_dev':before,'reverse_dev_diagnostic':metrics(reverse_dev),'pooled_dev_raw':after,'pooled_dev_calibrated':metrics(pooled_dev,fitted['temperature']),'pooled_dev_slices':slice_metrics(pooled_dev,fitted['temperature']),'pooled_calibration_raw':metrics(pooled_cal),'pooled_calibration_fitted':metrics(pooled_cal,fitted['temperature']),'gate':gate}
  write_json(out/'development.json',development);write_json(out/'dev-gate.json',gate);wb.log(development);checkpoint()
  def unchanged():return all(not p.requires_grad and p.grad is None and p._version==v for p,v in frozen)
  if gate['status']!='passed':
   if not unchanged():raise RuntimeError('Frozen model changed')
   write_json(out/'completion.json',{'status':'rejected_on_dev','attempted':0,'parameters_unchanged':True,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-started,'test_logits_read':False,'test_inference':False,'test_scored':False})
   return {'status':'rejected_on_dev','dev_gate':gate,'wandb':wb.url}
  # No benchmark rows or original test logits are read before this point.
  rows=load_panel();reverse_test=infer('test',rows);pooled_test=pool(original('test'),reverse_test);originals={r['id']:r for r in rows}
  for item in pooled_test:
   row=originals[item['id']];z=torch.tensor(item['logits'],dtype=torch.float64)
   for name,temp in [('raw',1.),('calibrated',fitted['temperature'])]:
    body=answer_body(row['question'],item['option_ids'],(z/temp).softmax(-1).tolist())
    value=record(row,dict(score(row,body),response=body,input_tokens=item['input_tokens'],latency_ms=item['latency_ms']))
    with (out/(name+'-predictions.jsonl')).open('a') as f:f.write(canonical(value)+'\n')
  if not unchanged():raise RuntimeError('Frozen model changed')
  write_json(out/'completion.json',{'status':'complete','attempted':len(rows),'parameters_unchanged':True,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-started})
  from report_trained_joint import report
  from report_prediction_dominance import summarize
  result=report(out,label='Frozen pretrained native/reverse probability pool');write_json(out/'prediction-distributions.json',summarize(out/'calibrated-predictions.jsonl'))
  wb.summary['test_composite']=result['macro_coverage_adjusted_accuracy'];wb.summary['test_per_suite']=result['suites']
  return {'status':'complete','dev_gate':gate,'composite':result['macro_coverage_adjusted_accuracy'],'wandb':wb.url}
 except Exception as exc:write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:
  if wb is not None:wb.finish()
  seal(out);checkpoint()
