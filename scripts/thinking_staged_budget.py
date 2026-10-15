"""Reservation checks by phase; unchanged measured2x timing safety and margins."""
import math
from thinking_smoke_plan import reservation_gate

def gate(capacity,stage,now,deadline,batch_size=4):
 if capacity.get('status')!='complete' or not capacity.get('capacity_validated') or not capacity.get('parameters_unchanged'):raise ValueError('Completed unchanged-parameter capacity smoke required')
 groups={'development':[857,809],'test':[2111]}
 if stage not in groups:raise ValueError('Unknown inference stage')
 if batch_size!=4:raise ValueError('Preregistered batch4 required')
 counts=groups[stage];padded=sum(math.ceil(n/batch_size)*batch_size for n in counts)
 result=reservation_gate(capacity['capacity_gate']['measured_batch_seconds'],now,deadline,records=padded,batch_size=batch_size)
 return dict(result,stage=stage,records=sum(counts),padded_records=padded,phase_counts=counts,policy='Each phase must fit remaining reservation; same2x measured max batch,600s margin,1200s idle; before-test gate follows unchanged devaccuracy gate.')
