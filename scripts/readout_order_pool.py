"""Equal arithmetic probability pool, with bijective semantic option alignment."""
import math

def _logsoftmax(values):
 if len(values)<2 or not all(math.isfinite(x) for x in values):raise ValueError('Require finite option logits')
 maximum=max(values);normalizer=maximum+math.log(sum(math.exp(x-maximum) for x in values))
 return [x-normalizer for x in values]

def pool(original,reversed_order):
 def indexed(rows):
  result={r['id']:r for r in rows}
  if len(rows)!=len(result):raise ValueError('Duplicate decision IDs')
  return result
 left,right=indexed(original),indexed(reversed_order)
 if not left or set(left)!=set(right):raise ValueError('Missing or unpaired decisions')
 result=[]
 for key in sorted(left):
  a,b=left[key],right[key];options=a['option_ids']
  if a['type']!=b['type'] or len(set(options))!=len(options) or len(set(b['option_ids']))!=len(b['option_ids']) or set(options)!=set(b['option_ids']):raise ValueError('Semantic option mismatch')
  if options!=list(reversed(b['option_ids'])):raise ValueError('Expected exactly reversed display')
  if len(a['logits'])!=len(options) or len(b['logits'])!=len(options):raise ValueError('Option/logit size differs')
  row={k:a[k] for k in ['id','type','option_ids']}
  if ('target_index' in a)!=('target_index' in b):raise ValueError('Label presence mismatch')
  if 'target_index' in a:
   if options[a['target_index']]!=b['option_ids'][b['target_index']]:raise ValueError('Semantic target mismatch')
   row['target_index']=a['target_index']
  if a.get('source')!=b.get('source'):raise ValueError('Source mismatch')
  if 'source' in a:row['source']=a['source']
  la=_logsoftmax(a['logits']);lb=dict(zip(b['option_ids'],_logsoftmax(b['logits'])))
  values=[]
  for option,x in zip(options,la):
   y=lb[option];m=max(x,y);values.append(m+math.log(math.exp(x-m)+math.exp(y-m))-math.log(2))
  row['logits']=values
  row['input_tokens']=a.get('input_tokens',0)+b.get('input_tokens',0)
  row['latency_ms']=a.get('latency_ms',0)+b.get('latency_ms',0)
  result.append(row)
 return result
