"""Dev-gated equal mixture of two locked, calibrated decision agents."""
import copy,hashlib,json,math
from pathlib import Path

def probabilities(logits,temperature=1.):
 if not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature')
 values=[x/temperature for x in logits]
 if not values or not all(math.isfinite(x) for x in values):raise ValueError('Invalid logits')
 maximum=max(values);e=[math.exp(x-maximum) for x in values];total=sum(e)
 return [x/total for x in e]

def mixture(head,readout,readout_temperature):
 def index(rows):
  result={r['id']:r for r in rows}
  if len(result)!=len(rows):raise ValueError('Duplicate IDs')
  return result
 left,right=index(head),index(readout)
 if not left or set(left)!=set(right):raise ValueError('Missing or unmatched IDs')
 result=[]
 for key in sorted(left):
  a,b=left[key],right[key];options=a['option_ids']
  if a['type']!=b['type'] or len(set(options))!=len(options) or len(set(b['option_ids']))!=len(b['option_ids']) or set(options)!=set(b['option_ids']):raise ValueError('Semantic option mismatch')
  pa=a['calibrated_probabilities'];pb=probabilities(b['logits'],readout_temperature)
  if len(pa)!=len(options) or len(pb)!=len(b['option_ids']) or not all(math.isfinite(x) and x>=0 for x in pa) or abs(sum(pa)-1)>1e-8:raise ValueError('Invalid component probabilities')
  aligned=dict(zip(b['option_ids'],pb));values=[.5*x+.5*aligned[o] for o,x in zip(options,pa)]
  if ('target_index' in a)!=('target_index' in b):raise ValueError('Gold presence mismatch')
  row={k:a[k] for k in ['id','type','option_ids']}
  if 'target_index' in a:
   if options[a['target_index']]!=b['option_ids'][b['target_index']]:raise ValueError('Semantic gold mismatch')
   row['target_index']=a['target_index']
  if a.get('source')!=b.get('source'):raise ValueError('Source mismatch')
  if 'source' in a:row['source']=a['source']
  row['logits']=[math.log(max(x,1e-300)) for x in values];result.append(row)
 return result

def accuracy(rows):
 return sum(max(range(len(r['logits'])),key=lambda i:r['logits'][i])==r['target_index'] for r in rows)/len(rows)

def verified_rows(root,name):
 root=Path(root);checks=json.loads((root/'checksums.json').read_text());raw=(root/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=checks[name]:raise ValueError('Changed source artifact: '+name)
 return [json.loads(x) for x in raw.decode().splitlines()]

def verified_json(root,name):
 root=Path(root);raw=(root/name).read_bytes();checks=json.loads((root/"checksums.json").read_text())
 if hashlib.sha256(raw).hexdigest()!=checks[name]:raise ValueError("Changed source artifact: "+name)
 return json.loads(raw)

def assess(head_root,readout_root):
 head_root,readout_root=Path(head_root),Path(readout_root)
 temp=verified_json(readout_root,'temperature.json')['temperature']
 hc=verified_json(head_root,'config.json');rc=verified_json(readout_root,'config.json')
 for role in ['dev','calibration']:
  if hc['manifest']['splits'][role]['points']['sha256']!=rc['manifest']['splits'][role]['points']['sha256']:raise ValueError('Component split bytes differ: '+role)
 head=verified_rows(head_root,'dev-traces.jsonl');readout=verified_rows(readout_root,'dev-logits.jsonl');mixed=mixture(head,readout,temp)
 expected=json.loads((head_root/'checksums.json').read_text())['best-head.safetensors']
 if len(mixed)!=809 or any(r['checkpoint_sha256']!=expected for r in head):raise ValueError('Wrong dev count or head checkpoint')
 head_acc=sum(max(range(len(r['calibrated_probabilities'])),key=lambda i:r['calibrated_probabilities'][i])==r['target_index'] for r in head)/len(head)
 lm_acc=accuracy(readout);mixed_acc=accuracy(mixed);stronger=max(head_acc,lm_acc)
 gate={'status':'passed' if mixed_acc-stronger>=.01-1e-12 else 'failed','head_accuracy':head_acc,'readout_accuracy':lm_acc,'mixture_accuracy':mixed_acc,'gain_vs_stronger_component':mixed_acc-stronger,'required_gain':.01,'dev_decisions':len(mixed)}
 return {'status':'dev_gate_'+gate['status'],'gate':gate,'type_slices':{t:{'decisions':sum(r['type']==t for r in mixed),'accuracy':accuracy([r for r in mixed if r['type']==t])} for t in sorted({r['type'] for r in mixed})},'source_slices':{s:{'decisions':sum(r['source']==s for r in mixed),'accuracy':accuracy([r for r in mixed if r['source']==s])} for s in sorted({r['source'] for r in mixed})},'test_read_or_scored':False,'components':{'head_checkpoint_sha256':expected,'readout_checkpoint_sha256':rc['checkpoint_sha256'],'dev_sha256':hc['manifest']['splits']['dev']['points']['sha256'],'calibration_sha256':hc['manifest']['splits']['calibration']['points']['sha256'],'readout_temperature':temp}}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('head_root');p.add_argument('readout_root');p.add_argument('output');a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Preserve prior experiment')
 result=assess(a.head_root,a.readout_root);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
