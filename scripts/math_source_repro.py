"""Strict matched-repeat gate for the head-only math attention runtime."""
import copy,json
from pathlib import Path

def runtime_config(source,run_id):
 cfg=copy.deepcopy(source)
 if cfg.get('selection_metric')!='accuracy':raise ValueError('Accuracy selection required')
 cfg.update(head_sdpa_backend='math',run_name=run_id)
 return cfg

def compare(left,right):
 left,right=Path(left),Path(right)
 def read(root,name):return json.loads((root/name).read_text())
 def rows(root,name):return [json.loads(x) for x in (root/name).read_text().splitlines()]
 checks={}
 for name in ['initial-head.safetensors','best-head.safetensors','best.json','temperature.json','dev-traces.jsonl','calibration-traces.jsonl']:
  checks[name]=(left/'training'/name).read_bytes()==(right/'training'/name).read_bytes()
 keys={'train_step':['event','step','epoch','loss','learning_rate','gradient_norm','decisions','record_ids']}
 def stable_metrics(root):
  out=[]
  for row in rows(root/'training','metrics.jsonl'):
   if row['event']=='train_step':out.append({k:row[k] for k in keys['train_step']})
   elif row['event'] in {'untrained_dev','dev_epoch'}:out.append({k:v for k,v in row.items() if k!='timestamp_unix'})
  return out
 a,b=stable_metrics(left),stable_metrics(right)
 checks['all_training_metrics']=a==b
 checks['510_updates_each']=sum(r['event']=='train_step' for r in a)==sum(r['event']=='train_step' for r in b)==510
 for name in ['raw-predictions.jsonl','calibrated-predictions.jsonl']:
  def stable(root):return [{k:v for k,v in r.items() if k!='latency_ms'} for r in rows(root/'test',name)]
  x,y=stable(left),stable(right)
  checks[name]=len(x)==len(y)==2111 and x==y
 return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,'scope':'Exact same-container, same-seed full training traces, selected checkpoint, calibration and all fixed-panel responses; no cross-hardware or cross-seed guarantee.'}
