"""Label-blind capacity plan for the fixed seven-example native readout."""
import copy

def select(prepared,maximum=12288,batch_size=4):
 if maximum!=12288 or batch_size!=4 or len(prepared)<8:raise ValueError('Fixed capacity plan requires eight real prompts and batch4')
 ordered=sorted(prepared,key=lambda r:(len(r['encoding']['input_ids']),r['id']))
 if any(not r['encoding']['input_ids'] or len(r['encoding']['input_ids'])>maximum for r in ordered):raise ValueError('Prompt outside capacity cap')
 chosen=[]
 for i in range(8):
  row=copy.deepcopy(ordered[round(i*(len(ordered)-1)/7)])
  chosen.append({k:row[k] for k in ['id','type','encoding']})
 longest=chosen[-1]
 for i in range(batch_size):
  row=copy.deepcopy(longest);row['id']='synthetic-fewshot-capacity-'+str(i);ids=row['encoding']['input_ids'];row['encoding']['input_ids']=(ids*((maximum+len(ids)-1)//len(ids)))[:maximum];chosen.append(row)
 return chosen
