"""Apply the fixed, dev-qualified mixture and locked calibration once."""
import copy,hashlib,json,time
from pathlib import Path
from evaluate_cached_mixture import mixture,probabilities,verified_rows,verified_json

def parent_probabilities(parent,option_ids):
 body=parent['response']['answers']['q']
 if body['type']=='noul':p={'true':body['noul'],'false':1-body['noul']}
 else:p=body['probabilities']
 if set(p)!=set(option_ids):raise ValueError('Parent semantic options differ')
 values=[p[k] for k in option_ids];total=sum(values)
 import math
 if any(not math.isfinite(x) or not 0<=x<=1 for x in values) or abs(total-1)>.025:raise ValueError('Parent probabilities fail original scorer contract')
 return [x/total for x in values]

def apply(head_training,head_test,readout,calibration,output):
 from compare_predictions import verified
 from run_baselines import load_panel,score,record,canonical
 from report_trained_joint import report
 from report_prediction_dominance import summarize
 from head_probe_io import write_json,seal
 started=time.monotonic();head_training,head_test,readout,output=map(Path,[head_training,head_test,readout,output]);fit=json.loads(Path(calibration).read_text());gate=fit['request']['dev_gate']
 if fit['status']!='complete' or gate['status']!='dev_gate_passed' or gate['gate']['gain_vs_stronger_component']<.01-1e-12:raise ValueError('Passed original gate and locked calibration required')
 if output.exists():raise FileExistsError('Preserve existing experiment output')
 temperature=fit['temperature']['temperature'];parents=gate['components'];hc=verified_json(head_test,'config.json');lc=verified_json(readout,'config.json')
 if hc['temperature']!=verified_json(head_training,'temperature.json')['temperature'] or parents['readout_temperature']!=verified_json(readout,'temperature.json')['temperature']:raise ValueError('Parent calibration differs')
 if hc['checkpoint_sha256']!=parents['head_checkpoint_sha256'] or lc['checkpoint_sha256']!=parents['readout_checkpoint_sha256'] or hc['panel_sha256']!=lc['panel_sha256']:raise ValueError('Parent checkpoint/panel differs')
 # Parameter fitting is finished. Only now load benchmark predictions and gold for scoring.
 head_rows=verified(head_test);verified(readout);lm_rows=verified_rows(readout,'test-logits.jsonl');head_index={r['id']:r for r in head_rows};typed=[];fallbacks=[]
 if set(head_index)!={r['id'] for r in lm_rows}:raise ValueError('Parent test IDs differ')
 for r in lm_rows:
  parent=head_index[r['id']]
  if parent['status']=='ok':
   body=parent['response']['answers']['q']
   if body['type']!=r['type']:raise ValueError('Invalid typed parent response')
   p=dict(zip(r['option_ids'],parent_probabilities(parent,r['option_ids'])))
  else:
   p=dict(zip(r['option_ids'],probabilities(r['logits'],parents['readout_temperature'])));fallbacks.append({'id':r['id'],'missing_component':'E5','reason':parent.get('detail',parent.get('error'))})
  typed.append({'id':r['id'],'type':r['type'],'option_ids':r['option_ids'],'calibrated_probabilities':[p[k] for k in r['option_ids']]})
 mixed=mixture(typed,lm_rows,parents['readout_temperature']);rows={r['id']:r for r in load_panel()};lm_index={r['id']:r for r in lm_rows}
 if len(mixed)!=2111 or set(rows)!={r['id'] for r in mixed}:raise ValueError('Incomplete fixed panel')
 protocol={'temperature':temperature,'component_weights':[.5,.5],'component_head_checkpoint_sha256':hc['checkpoint_sha256'],'component_readout_checkpoint_sha256':lc['checkpoint_sha256'],'panel_sha256':lc['panel_sha256'],'calibration':fit,'selection':'Original fixed equal mixture passed>=1pp dev gain over stronger component; no test fitting','probability_method':'Arithmetic mean of separately calibration-scaled semantic probabilities, followed by one calibration-only globalT','latency_definition':'Sum of parent head and original readout forward latency; cached CPU application time is not deployment inference latency','missing_component_rule':'Use available component, no example deletion; same finalT','fallbacks':fallbacks,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 protocol['checkpoint_sha256']=hashlib.sha256(canonical(protocol).encode()).hexdigest();output.mkdir(parents=True);write_json(output/'config.json',protocol);write_json(output/'temperature.json',fit['temperature'])
 for item in mixed:
  row=rows[item['id']];parent=head_index[item['id']];lm=lm_index[item['id']]
  for name,temp in [('raw',1.),('calibrated',temperature)]:
   p=dict(zip(item['option_ids'],probabilities(item['logits'],temp)));answer={'type':item['type']}
   if item['type']=='noul':answer['noul']=p['true']
   else:answer['probabilities']=p
   body={'answers':{'q':answer}};scored=record(row,dict(score(row,body),response=body,input_tokens=parent.get('input_tokens',0)+lm.get('input_tokens',0),latency_ms=parent.get('latency_ms',0)+lm['latency_ms']))
   with (output/(name+'-predictions.jsonl')).open('a') as stream:stream.write(canonical(scored)+'\n')
 write_json(output/'completion.json',{'status':'complete','attempted':len(mixed),'parameters_unchanged':True,'optimizer_updates':0,'new_GPU_forward_passes':0,'elapsed_seconds':time.monotonic()-started,'timing_scope':'CPU-only application/scoring of existing verified inference outputs'})
 metrics=report(output,label='E5 / frozen pretrained equal probability mixture');write_json(output/'prediction-distributions.json',summarize(output/'calibrated-predictions.jsonl'));seal(output)
 return metrics
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('head_training');p.add_argument('head_test');p.add_argument('readout');p.add_argument('calibration');p.add_argument('output');a=p.parse_args();apply(a.head_training,a.head_test,a.readout,a.calibration,a.output)
