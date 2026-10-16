"""Fixed-policy dev assessment; temperature fitting and test execution stay separate."""
from readout_order_pool import pool
from selective_thinking import combine,routed_ids

def assess(native_cal,reverse_cal,thinking_cal,native_dev,reverse_dev,thinking_dev):
 cal=combine(native_cal,reverse_cal,thinking_cal);dev=combine(native_dev,reverse_dev,thinking_dev)
 if set(x['id'] for x in cal)&set(x['id'] for x in dev):raise ValueError('Calibration/development overlap')
 def accuracy(rows):
  return sum(max(range(len(r['logits'])),key=lambda i:r['logits'][i])==r['target_index'] for r in rows)/len(rows)
 before=accuracy(pool(native_dev,reverse_dev));after=accuracy(dev);gain=after-before
 gate={'status':'passed' if gain>=.01-1e-12 else 'failed','pooled_dev_accuracy':before,'selective_dev_accuracy':after,'observed_gain':gain,'required_gain':.01,'dev_decisions':len(dev)}
 return {'status':'dev_gate_'+gate['status'],'gate':gate,'routing':{'calibration':len(routed_ids(native_cal,reverse_cal)),'dev':len(routed_ids(native_dev,reverse_dev))},'test_read':False,'selection_metric':'accuracy'},cal,dev
