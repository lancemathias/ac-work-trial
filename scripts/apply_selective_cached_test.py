"""Apply the fixed selective-thinking policy only after its own dev gate."""
import hashlib,json,math,time
from pathlib import Path

def require_development(fit):
 gate=fit.get('gate',{});before=gate.get('pooled_dev_accuracy');after=gate.get('selective_dev_accuracy');gain=gate.get('observed_gain');t=fit.get('temperature',{}).get('temperature')
 if fit.get('status')!='dev_gate_passed' or gate.get('status')!='passed' or fit.get('selection_metric')!='accuracy' or gate.get('selection_metric','accuracy')!='accuracy':raise ValueError('Passed accuracy development gate required')
 if not all(type(x) in [int,float] and math.isfinite(x) for x in [before,after,gain,t]):raise ValueError('Finite dev statistics and temperature required')
 if gate.get('required_gain')!=.01 or gate.get('dev_decisions')!=809 or gain<.01-1e-12 or abs((after-before)-gain)>1e-12:raise ValueError('Original one-point dev threshold required')
 if abs(before-0.8430160692212608)>1e-12 or not 0<=after<=1 or t<=0:raise ValueError('Wrong pooled comparator or temperature')
 return t

def apply(development_root,output):
 from evaluate_cached_mixture import verified_rows,verified_json,probabilities
 from selective_thinking import combine,routed_ids
 from training_data import digest,canonical
 from run_baselines import ROOT,load_panel,record,score
 from head_probe_io import write_json,seal
 from report_trained_joint import report
 from report_prediction_dominance import summarize
 development_root=Path(development_root);output=Path(output);started=time.monotonic()
 fit=verified_json(development_root,'result.json');temperature=require_development(fit)
 if output.exists():raise FileExistsError('Preserve existing experiment evidence')
 native=Path('/results/readout-fast-qwen38-27b-v1');reverse=Path('/results/readout-native-reverse-pool-v1');thinking=Path('/results/readout-bounded-thinking128-staged-v2/attempts')/fit['request']['thinking_attempt']
 if thinking.parent!=Path('/results/readout-bounded-thinking128-staged-v2/attempts') or not thinking.name.isalnum():raise ValueError('Invalid recorded thinking attempt')
 for root,key in [(native,'native_checksums_sha256'),(reverse,'pool_checksums_sha256')]:
  if digest(root/'checksums.json')!=fit['request'][key]:raise ValueError('Parent manifest changed')
 for name,expected in fit['request']['source_sha256'].items():
  if digest(ROOT/name)!=expected:raise ValueError('Assessed policy source changed: '+name)
 for name,expected in fit['request']['input_sha256'].items():
  if digest(name)!=expected:raise ValueError('Assessed thinking development changed')
 # Original thinking must be complete and parameter-verified before reusing test logits.
 completion=verified_json(thinking,'completion.json')
 if completion['status']!='complete' or completion['attempted']!=2111 or completion['parameters_unchanged'] is not True or completion['optimizer_updates']!=0:raise ValueError('Complete frozen thinking test parent required for cached application')
 cfg=verified_json(native,'config.json');tc=verified_json(thinking,'config.json')
 if tc['parent_checkpoint_sha256']!=cfg['checkpoint_sha256'] or tc['panel_sha256']!=cfg['panel_sha256']:raise ValueError('Thinking parent differs')
 # Only now read benchmark predictions. No fitting occurs after this point.
 a=verified_rows(native,'test-logits.jsonl');b=verified_rows(reverse,'reverse-test-logits.jsonl');c=verified_rows(thinking,'test-logits.jsonl');items=combine(a,b,c)
 rows={r['id']:r for r in load_panel()}
 if len(items)!=2111 or {r['id'] for r in items}!=set(rows):raise ValueError('Incomplete fixed test panel')
 protocol={'kind':'fixed_order_disagreement_thinking128','temperature':temperature,'selection':'Fixed accuracy gain>=1pp over pooled84.3016069% dev; no test fitting','development':fit,'panel_sha256':cfg['panel_sha256'],'thinking_routed':len(routed_ids(a,b)),'new_GPU_forward_passes':0,'latency_definition':'Native+reverse forwards for every decision plus thinking generation/final forward only for disagreement; amortized parent compute excludes setup/CPU pooling','runner_sha256':digest(__file__)};protocol['checkpoint_sha256']=hashlib.sha256(canonical(protocol).encode()).hexdigest();output.mkdir(parents=True);write_json(output/'config.json',protocol);write_json(output/'temperature.json',fit['temperature'])
 for item in items:
  row=rows[item['id']]
  for name,temp in [('raw',1.),('calibrated',temperature)]:
   p=dict(zip(item['option_ids'],probabilities(item['logits'],temp)));answer={'type':item['type']}
   if item['type']=='noul':answer['noul']=p['true']
   else:answer['probabilities']=p
   body={'answers':{'q':answer}};value=record(row,dict(score(row,body),response=body,input_tokens=item['input_tokens'],latency_ms=item['latency_ms']))
   with (output/(name+'-predictions.jsonl')).open('a') as f:f.write(canonical(value)+'\n')
 write_json(output/'completion.json',{'status':'complete','attempted':2111,'parameters_unchanged':True,'optimizer_updates':0,'new_GPU_forward_passes':0,'elapsed_seconds':time.monotonic()-started})
 result=report(output,label='Native/reverse disagreement thinking128');write_json(output/'prediction-distributions.json',summarize(output/'calibrated-predictions.jsonl'));seal(output);return result
