"""One bounded Boolean intercept fitted solely from calibration labels."""
import copy,math
GRID=tuple(i/8 for i in range(-32,33))
def shifted(item,bias):
 if not math.isfinite(bias):raise ValueError('Nonfinite offset')
 result=copy.deepcopy(item)
 if result['type']=='noul':
  if len(result['option_ids'])!=2 or set(result['option_ids'])!={'true','false'}:raise ValueError('Require semantic Boolean IDs')
  result['logits'][result['option_ids'].index('true')]+=bias
 return result

def accuracy(items,bias=0):
 if not items:raise ValueError('Empty labelled set')
 return sum(max(range(len(r['logits'])),key=lambda i:r['logits'][i])==r['target_index'] for r in [shifted(x,bias) for x in items])/len(items)

def fit(items,role):
 if role!='calibration':raise ValueError('Only calibration labels may fit offset')
 boolean=[r for r in items if r['type']=='noul']
 if not boolean:raise ValueError('No Boolean calibration examples')
 for r in boolean:
  if type(r.get('target_index')) is not int or r['target_index'] not in [0,1]:raise ValueError('Missing Boolean calibration label')
 scores={b:accuracy(boolean,b) for b in GRID};bias=min(GRID,key=lambda b:(-scores[b],abs(b),b))
 return {'bias':bias,'calibration_boolean_decisions':len(boolean),'calibration_boolean_accuracy_before':scores[0.0],'calibration_boolean_accuracy_after':scores[bias],'grid_min':-4,'grid_max':4,'grid_step':.125,'objective':'accuracy','tie_break':'smallest absolute offset, then smaller signed offset'}

def dev_gate(items,bias):
 boolean=[r for r in items if r['type']=='noul'];before=accuracy(items);after=accuracy(items,bias);bb=accuracy(boolean);ba=accuracy(boolean,bias)
 return {'status':'passed' if after-before>=.01-1e-12 and ba-bb>=.05-1e-12 else 'failed','dev_accuracy_before':before,'dev_accuracy_after':after,'boolean_accuracy_before':bb,'boolean_accuracy_after':ba,'dev_decisions':len(items),'boolean_decisions':len(boolean)}
