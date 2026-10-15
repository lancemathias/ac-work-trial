"""Read experiment summaries using the established Modal W&B secret."""
import json
from pathlib import Path
import modal
ENV='worktrial-lance';ROOT=Path(__file__).resolve().parents[1]
app=modal.App('ac-worktrial-research-inspection')
image=modal.Image.debian_slim(python_version='3.12').pip_install('wandb==0.22.3')
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
@app.function(image=image,cpu=1,memory=1024,timeout=300,volumes={'/results':results.read_only()},secrets=[secret])
def inspect():
 import wandb
 rows=[]
 for run in wandb.Api().runs('lmathias/ac-worktrial-decisions',order='-created_at'):
  rows.append({'id':run.id,'name':run.name,'state':run.state,'url':run.url,'summary':dict(run.summary),'config':run.config})
  if len(rows)>=12:break
 files={}
 for d in Path('/results').glob('e[23]*'):
  if d.is_dir():
   files[d.name]={}
   for name in ['summary.json','completion.json','best.json','temperature.json','wandb-run.json']:
    if (d/name).exists():files[d.name][name]=json.loads((d/name).read_text())
 return {'wandb':rows,'files':files}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=inspect.remote();p=ROOT/'outputs/autoresearch/existing-runs.json';p.write_text(json.dumps(result,indent=2,default=str)+'\n')
 print(json.dumps([{'id':r['id'],'name':r['name'],'state':r['state'],'summary':r['summary']} for r in result['wandb']],indent=2,default=str))
