"""Label-blind smoke selection and conservative reservation gate for thinking."""
import math

def select(prepared, count=8):
 if count < 2 or not prepared:
  raise ValueError('At least two quantiles and nonempty inputs required')
 ordered=sorted(prepared,key=lambda row:(len(row['encoding']['input_ids']),row['id']))
 indexes=sorted({round(i*(len(ordered)-1)/(count-1)) for i in range(count)})
 # Never expose development labels/source metadata to generation or selection.
 return [{'id':ordered[i]['id'],'type':ordered[i]['type'],'encoding':ordered[i]['encoding']} for i in indexes]

def reservation_gate(batch_seconds, now, deadline, records=3777, batch_size=4,
                     idle_seconds=1200, margin_seconds=600, safety_factor=2.):
 values=list(batch_seconds)
 if not values or any(not math.isfinite(x) or x <= 0 for x in values):
  raise ValueError('Positive finite measured batch times required')
 if batch_size <= 0 or records <= 0 or safety_factor < 1:
  raise ValueError('Invalid capacity assumptions')
 # Estimate only: the longest held-out input and unseen kernel shapes can cost more.
 estimate=max(values)*math.ceil(records/batch_size)*safety_factor
 required=estimate+idle_seconds+margin_seconds
 return {'status':'passed' if now+required < deadline else 'insufficient_reservation',
         'measured_batch_seconds':values,'estimated_trial_seconds':estimate,
         'required_remaining_seconds':required,'remaining_seconds':deadline-now,
         'records':records,'batch_size':batch_size,'safety_factor':safety_factor,
         'qualification':'Timing estimate, not a guarantee; full runner must enforce deadline between batches. No accuracy-based smoke selection.'}
