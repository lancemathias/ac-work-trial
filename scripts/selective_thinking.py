"""Label-independent native/reverse disagreement router with fixed thinking budget."""
import math
from readout_order_pool import pool,_logsoftmax

def winner(row):
 options=row['option_ids'];values=row['logits']
 if len(options)!=len(values) or len(set(options))!=len(options) or not values or not all(math.isfinite(x) for x in values):raise ValueError('Invalid routing logits')
 return min(options,key=lambda key:(-values[options.index(key)],key))

def routed_ids(native,reverse):
 # Pool validates pairing and semantic alignment; labels never enter the rule.
 pool(native,reverse);right={r['id']:r for r in reverse}
 return sorted(r['id'] for r in native if winner(r)!=winner(right[r['id']]))

def combine(native,reverse,thinking):
 pooled=pool(native,reverse);wanted=set(routed_ids(native,reverse));known={r['id'] for r in pooled};lookup={r['id']:r for r in thinking}
 if len(lookup)!=len(thinking) or not wanted.issubset(lookup) or not set(lookup).issubset(known):raise ValueError('Missing, duplicate or foreign thinking decisions')
 result=[]
 for base in pooled:
  row=dict(base);key=row['id'];row['thinking_routed']=key in wanted
  if key in wanted:
   candidate=lookup[key];options=row['option_ids'];other=candidate['option_ids']
   if candidate['type']!=row['type'] or len(set(other))!=len(other) or set(other)!=set(options) or len(candidate['logits'])!=len(other):raise ValueError('Thinking semantic options differ')
   if ('target_index' in base)!=('target_index' in candidate):raise ValueError('Target presence differs')
   if 'target_index' in base and options[base['target_index']]!=other[candidate['target_index']]:raise ValueError('Thinking target differs')
   if candidate.get('source')!=base.get('source'):raise ValueError('Thinking source differs')
   values=dict(zip(other,_logsoftmax(candidate['logits'])));row['logits']=[values[o] for o in options]
   row['input_tokens']+=candidate.get('expanded_tokens',candidate.get('input_tokens',0));row['latency_ms']+=candidate.get('latency_ms',0)
  result.append(row)
 return result
