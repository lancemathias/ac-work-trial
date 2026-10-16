"""Explicit development-only checkpoint selection; earlier epoch wins ties."""
def selection_value(dev,metric):
 if metric=='accuracy':return -float(dev['accuracy'])
 if metric=='nll':return float(dev['nll'])
 raise ValueError('Unsupported checkpoint selection metric: '+metric)
def best_epoch(events,metric='accuracy'):
 rows=[r for r in events if r.get('event')=='dev_epoch']
 if not rows:raise ValueError('No development epoch records')
 return min(rows,key=lambda r:(selection_value(r,metric),r['epoch']))
