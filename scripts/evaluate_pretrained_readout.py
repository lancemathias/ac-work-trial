"""Fixed pretrained answer-letter readout; calibration is the only fitting."""
import hashlib,json,time
from pathlib import Path

def prepare_decisions(tokenizer,rows,training_sources=False):
 from pretrained_readout import encode
 result=[]
 for row in rows:
  if training_sources:
   request=row['request'];question=request['questions']['q'];state=request['state']
  else:question=row['question'];state=row['state']
  encoding=encode(tokenizer,state,question)
  item={'id':row['id'],'type':question['type'],'encoding':encoding}
  if training_sources:item.update(source=row['source'],target_index=encoding['option_ids'].index(row['targets']['q']))
  result.append(item)
 return result

def evaluate(output,commit,checkpoint=lambda:None):
 import torch,wandb
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from pretrained_readout import predict,SYSTEM_PROMPT
 from kernel_runtime import configure_optimized_runtime
 from training_data import load_splits,digest,canonical
 from train_joint import fit_temperature,metrics,slice_metrics,write_json
 from run_baselines import ROOT,load_panel,record,score
 from evaluate_trained_joint import answer_body
 out=Path(output);out.mkdir(exist_ok=False);started=time.monotonic();wb=None
 cfg=json.loads((ROOT/'configs/e4-qwen38-27b-v3.json').read_text())
 splits,manifest=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json')
 runtime=configure_optimized_runtime('both')
 protocol={'backbone_id':cfg['backbone_id'],'backbone_revision':cfg['backbone_revision'],
  'system_prompt':SYSTEM_PROMPT,'codes':'ABCDEFGHIJKLMNOPQRSTUVWXYZ','thinking':False,'attention_mode':'native_causal',
  'temperature_role':'existing nonbenchmark v3 calibration only','selection':'fixed prompt and pretrained weights; no checkpoint selection',
  'kernel_runtime':runtime,'seed':cfg['seed'],'max_input_tokens':8192,'batch_size':4,
  'latency_definition':'amortized per-record batch inference, excluding prompt encoding and scoring',
  'commit':commit,'runner_sha256':digest(__file__),'readout_sha256':digest(ROOT/'scripts/pretrained_readout.py'),
  'panel_sha256':digest(ROOT/'data/manifests/baseline-panel.json'),
  'manifest':manifest,'tokenizer_preflight_sha256':digest('/project/readout-preflight.json'),'checkpoint_identity_note':'SHA256 of frozen model revision and readout protocol; no trained head checkpoint'}
 protocol['checkpoint_sha256']=hashlib.sha256(canonical(protocol).encode()).hexdigest()
 write_json(out/'config.json',protocol);checkpoint()
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 try:
  wb=wandb.init(project='ac-worktrial-decisions',name=out.name,config=protocol,job_type='pretrained-readout')
  write_json(out/'wandb-run.json',{'id':wb.id,'url':wb.url})
  processor=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True)
  model=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'}).requires_grad_(False).eval()
  frozen=[(p,p._version) for p in model.parameters()]
  def infer(role,prepared,extra_emit=lambda item:None):
   expected=json.loads(Path('/project/readout-preflight.json').read_text())['sets'][role]
   fingerprint=hashlib.sha256()
   for item in prepared:fingerprint.update(json.dumps({'id':item['id'],**item['encoding']},sort_keys=True,separators=(',',':')).encode())
   if len(prepared)!=expected['expected'] or fingerprint.hexdigest()!=expected['encodings_sha256']:raise ValueError('Locked tokenizer preflight differs: '+role)
   path=out/(role+'-logits.jsonl');count=0
   def emit(item):
    nonlocal count
    with path.open('a') as stream:stream.write(canonical(item)+'\n')
    extra_emit(item);count+=1
    if count%100==0:checkpoint()
   result=predict(model,processor.tokenizer,prepared,batch_size=4,emit=emit)
   if len(result)!=len(prepared) or len({x['id'] for x in result})!=len(prepared):raise ValueError('Missing or duplicate predictions')
   checkpoint();return result
  calibration=infer('calibration',prepare_decisions(processor.tokenizer,splits['calibration'],True))
  fitted=fit_temperature(calibration);write_json(out/'temperature.json',fitted)
  protocol['temperature']=fitted['temperature'];write_json(out/'config.json',protocol);checkpoint()
  dev=infer('dev',prepare_decisions(processor.tokenizer,splits['dev'],True))
  development={'dev_raw':metrics(dev),'dev_calibrated':metrics(dev,fitted['temperature']),
   'dev_slices':slice_metrics(dev,fitted['temperature']),'calibration_raw':metrics(calibration),
   'calibration_fitted':metrics(calibration,fitted['temperature'])}
  write_json(out/'development.json',development);wb.log(development);checkpoint()
  # Benchmark gold is available only to the scorer; it never enters prepared model inputs.
  rows=load_panel();originals={r['id']:r for r in rows}
  def emit_test(item):
   row=originals[item['id']];logits=torch.tensor(item['logits'],dtype=torch.float64)
   for name,temp in [('raw',1.),('calibrated',fitted['temperature'])]:
    body=answer_body(row['question'],item['option_ids'],(logits/temp).softmax(-1).tolist())
    value=record(row,dict(score(row,body),response=body,input_tokens=item['input_tokens'],latency_ms=item['latency_ms']))
    with (out/(name+'-predictions.jsonl')).open('a') as stream:stream.write(canonical(value)+'\n')
  infer('test',prepare_decisions(processor.tokenizer,rows),emit_test)
  unchanged=all(not p.requires_grad and p.grad is None and p._version==v for p,v in frozen)
  if not unchanged:raise RuntimeError('Readout modified frozen parameters')
  write_json(out/'completion.json',{'status':'complete','attempted':len(rows),'parameters_unchanged':True,'elapsed_seconds':time.monotonic()-started,'optimizer_updates':0})
  from report_trained_joint import report
  result=report(out,label='Frozen pretrained Qwen3.8-27B letter readout')
  from report_prediction_dominance import summarize
  write_json(out/'prediction-distributions.json',summarize(out/'calibrated-predictions.jsonl'))
  wb.summary['test_composite']=result['macro_coverage_adjusted_accuracy'];wb.summary['test_per_suite']=result['suites']
  return {'status':'complete','run_id':out.name,'composite':result['macro_coverage_adjusted_accuracy'],'wandb':wb.url}
 except Exception as exc:
  write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:
  if wb is not None:wb.finish()
  write_json(out/'checksums.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='checksums.json'});checkpoint()
