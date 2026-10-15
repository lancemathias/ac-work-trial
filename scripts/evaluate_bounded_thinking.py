"""Fixed-budget thinking trial with nonbenchmark accuracy gate before test access."""
import hashlib,json,time,uuid
from pathlib import Path

def evaluate(output,commit,deadline,checkpoint=lambda:None):
 from run_baselines import ROOT
 from training_data import digest
 from head_probe_io import prepare_run,write_json,seal
 proof_path=Path('/project/bounded-thinking-cpu.json');proof=json.loads(proof_path.read_text())
 if proof['status']!='passed':raise ValueError('CPU generation contracts required')
 for name,expected in proof['source_sha256'].items():
  if digest(ROOT/name)!=expected:raise ValueError('CPU-tested source changed: '+name)
 baseline=Path('/results/readout-fast-qwen38-27b-v1');root=Path(output)
 request={'kind':'fixed128_native_thinking','commit':commit,'runner_sha256':digest(__file__),'cpu_proof_sha256':digest(proof_path),'baseline_checksums_sha256':digest(baseline/'checksums.json')}
 prior=prepare_run(root,request);checkpoint()
 if prior is not None:return prior
 attempts=root/'attempts';attempts.mkdir(exist_ok=True)
 if len(list(attempts.glob('*-started.json')))>=3:raise RuntimeError('Three interrupted attempts; diagnose before resubmission')
 identity=uuid.uuid4().hex;out=attempts/identity
 write_json(attempts/(identity+'-started.json'),{'request':request,'time':time.time()});checkpoint()
 try:
  value=_evaluate_once(out,commit,deadline,checkpoint,baseline,proof,root.name)
  value=dict(value,request=request,attempt_id=identity,run_id=root.name)
  write_json(root/'result.json',value);seal(root);checkpoint();return value
 finally:checkpoint()

def _evaluate_once(out,commit,deadline,checkpoint,baseline,proof,run_name):
 import torch,wandb
 from run_baselines import ROOT,load_panel,record,score
 from training_data import digest,load_splits,canonical
 from head_probe_io import write_json,seal
 from train_joint import metrics,fit_temperature,slice_metrics
 from resident_backbone import get_processor,get_model
 from kernel_runtime import configure_optimized_runtime
 from bounded_thinking_readout import prepare,predict
 from evaluate_trained_joint import answer_body
 out=Path(out);out.mkdir(exist_ok=False);baseline=Path(baseline);began=time.monotonic();wb=None
 def window():
  if time.time()+600+1200+60>=deadline:raise TimeoutError('Stop before next batch: reservation includes batch margin and idle expiry')
 window()
 checks=json.loads((baseline/'checksums.json').read_text())
 def verified(name,lines=False):
  path=baseline/name
  if digest(path)!=checks[name]:raise ValueError('Baseline artifact changed: '+name)
  return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
 old=verified('config.json');completed=verified('completion.json')
 if completed['status']!='complete' or not completed['parameters_unchanged']:raise ValueError('Complete frozen baseline required')
 settings=json.loads((ROOT/'configs/bounded-thinking128.json').read_text());cfg=json.loads((ROOT/'configs/e4-qwen38-27b-v3.json').read_text())
 if any(old[k]!=cfg[k] or settings[k]!=cfg[k] for k in ['backbone_id','backbone_revision']):raise ValueError('Backbone mismatch')
 runtime=configure_optimized_runtime('both')
 protocol={'settings':settings,'kernel_runtime':runtime,'commit':commit,'runner_sha256':digest(__file__),'selection':'same809 nonbenchmark dev accuracy gain>=1pp over native readout','temperature_role':'same857 nonbenchmark calibration only','panel_sha256':old['panel_sha256'],'parent_checkpoint_sha256':old['checkpoint_sha256'],'attention_mode':'native_causal','optimizer_updates':0,'latency_definition':'Thinking generation plus final full prefill, amortized per batch; excludes model load and input encoding'}
 protocol['checkpoint_sha256']=hashlib.sha256(canonical(protocol).encode()).hexdigest();write_json(out/'config.json',protocol)
 splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json')
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 try:
  wb=wandb.init(project='ac-worktrial-decisions',name=run_name,job_type='bounded-thinking',config=protocol);write_json(out/'wandb-run.json',{'id':wb.id,'url':wb.url});checkpoint()
  tokenizer=get_processor(cfg).tokenizer;model=get_model(cfg);frozen=[(p,p._version) for p in model.parameters()]
  def infer(role,rows):
   prepared=prepare(tokenizer,rows,settings,training_sources=role!='test');h=hashlib.sha256()
   for item in prepared:h.update(json.dumps({'id':item['id'],**item['encoding']},sort_keys=True,separators=(',',':')).encode())
   expected=proof['sets'][role]
   if len(prepared)!=expected['records'] or h.hexdigest()!=expected['encodings_sha256']:raise ValueError('Prepared input differs from CPU proof: '+role)
   ordered=sorted(prepared,key=lambda x:(len(x['encoding']['input_ids']),x['id']));result=[]
   def emit(item):
    with (out/(role+'-logits.jsonl')).open('a') as f:f.write(canonical(item)+'\n')
   for begin in range(0,len(ordered),settings['batch_size']):
    window();result.extend(predict(model,tokenizer,ordered[begin:begin+settings['batch_size']],settings,emit=emit));checkpoint()
   if len(result)!=len(prepared) or len({x['id'] for x in result})!=len(result):raise ValueError('Incomplete thinking predictions')
   return result
  cal=infer('calibration',splits['calibration']);fitted=fit_temperature(cal);write_json(out/'temperature.json',fitted)
  dev=infer('dev',splits['dev']);before=metrics(verified('dev-logits.jsonl',True));after=metrics(dev)
  gain=after['accuracy']-before['accuracy'];gate={'status':'passed' if gain>=settings['dev_advance_minimum_gain']-1e-12 else 'failed','accuracy_before':before['accuracy'],'accuracy_after':after['accuracy'],'observed_gain':gain,'required_gain':settings['dev_advance_minimum_gain']}
  development={'native_dev':before,'thinking_dev_raw':after,'thinking_dev_calibrated':metrics(dev,fitted['temperature']),'dev_slices':slice_metrics(dev,fitted['temperature']),'gate':gate};write_json(out/'development.json',development);write_json(out/'dev-gate.json',gate);wb.log(development);checkpoint()
  def unchanged():return all(not p.requires_grad and p.grad is None and p._version==v for p,v in frozen)
  if gate['status']!='passed':
   if not unchanged():raise RuntimeError('Frozen parameters changed')
   write_json(out/'completion.json',{'status':'rejected_on_dev','attempted':0,'parameters_unchanged':True,'optimizer_updates':0,'test_read':False,'test_inference':False,'test_scored':False,'elapsed_seconds':time.monotonic()-began})
   return {'status':'rejected_on_dev','dev_gate':gate,'wandb':wb.url}
  rows=load_panel();predictions=infer('test',rows);originals={r['id']:r for r in rows}
  for item in predictions:
   row=originals[item['id']];z=torch.tensor(item['logits'],dtype=torch.float64)
   for name,temp in [('raw',1.),('calibrated',fitted['temperature'])]:
    body=answer_body(row['question'],item['option_ids'],(z/temp).softmax(-1).tolist());value=record(row,dict(score(row,body),response=body,input_tokens=item['expanded_tokens'],latency_ms=item['latency_ms']))
    with (out/(name+'-predictions.jsonl')).open('a') as f:f.write(canonical(value)+'\n')
  if not unchanged():raise RuntimeError('Frozen parameters changed')
  write_json(out/'completion.json',{'status':'complete','attempted':len(rows),'parameters_unchanged':True,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-began})
  from report_trained_joint import report
  from report_prediction_dominance import summarize
  result=report(out,label='Frozen bounded thinking128');write_json(out/'prediction-distributions.json',summarize(out/'calibrated-predictions.jsonl'));wb.summary['test_composite']=result['macro_coverage_adjusted_accuracy'];wb.summary['test_per_suite']=result['suites']
  return {'status':'complete','dev_gate':gate,'composite':result['macro_coverage_adjusted_accuracy'],'wandb':wb.url}
 except Exception as exc:write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:
  if wb is not None:wb.finish()
  seal(out);checkpoint()
