"""Export small live evidence snapshots without credentials or input text."""
import json,datetime
from pathlib import Path
import modal
from modal.config import _set_profile
_set_profile('ac-worktrial')
ROOT=Path(__file__).resolve().parents[1]
RUNS=['e3-medium-v3-seed20261014-e7dd4bb','e4-qwen38-27b-v3-seed20261014-2d9ef9a']
v=modal.Volume.from_name('ac-worktrial-training-results',environment_name='worktrial-lance')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=ROOT/'outputs/autoresearch/snapshots'/stamp;out.mkdir(parents=True,exist_ok=False)
for run_id in RUNS:
 summary={}
 for suffix,names in [('', ['summary.json','completion.json','wandb-run.json']),('-test',['metrics.json','completion.json','pipeline.json']),('-pipeline',['status.json','cpu-validation.json'])]:
  for name in names:
   remote=run_id+suffix+'/'+name
   try:raw=b''.join(v.read_file(remote))
   except (modal.exception.NotFoundError,FileNotFoundError):continue
   path=out/(run_id+suffix)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
   data=json.loads(raw)
   if name=='metrics.json':data={k:v for k,v in data.items() if k!='suites'}
   if name=='summary.json':data={k:data[k] for k in ['status','best_dev_nll','dev_calibrated','temperature'] if k in data}
   if name=='cpu-validation.json':data={k:(x[-2000:] if isinstance(x,str) else x) for k,x in data.items()}
   summary[suffix+'/'+name]=data
 print(run_id,json.dumps(summary,indent=2))
print('snapshot',out)
