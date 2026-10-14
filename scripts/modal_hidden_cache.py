"""Cache the exact frozen E1 backbone features in the authorized Modal environment."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
ENVIRONMENT='worktrial-lance'
if modal.is_local():
 from modal.config import config,_profile
 if _profile!='ac-worktrial' or config.get('environment')!=ENVIRONMENT:raise ValueError('Use --profile ac-worktrial --env worktrial-lance')
app=modal.App('ac-worktrial-hidden-cache')
inspect_image=modal.Image.debian_slim(python_version='3.12').pip_install('wandb==0.22.3')
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENVIRONMENT,required_keys=['WANDB_API_KEY'])
@app.function(image=inspect_image,secrets=[secret],timeout=180)
def inspect_run():
 import wandb
 api=wandb.Api();runs=list(api.runs('lmathias/ac-worktrial-decisions',order='-created_at',per_page=5))[:5]
 return [{'id':r.id,'name':r.name,'url':r.url,'state':r.state,'config':r.config,'summary':dict(r.summary)} for r in runs]
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install_from_requirements(ROOT/'requirements-hidden-cache.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts')
 .add_local_dir(ROOT/'configs','/project/configs'))
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENVIRONMENT)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENVIRONMENT)
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=7200,max_containers=1,
 volumes={'/data':data,'/models':models})
def build_cache():
 import sys
 sys.path.insert(0,'/project/scripts')
 from cache_hidden_runtime import extract
 snapshot=json.loads(Path('/project/configs/hidden-cache-nulun3tv.json').read_text())
 cfg=snapshot['config']['config']
 out=Path('/data/hidden-cache/nulun3tv-qwen35-9b-3b3d8db82da7')
 try:
  return extract(snapshot,Path('/data')/cfg['dataset']['path'].lstrip('/'),
   '/project/configs/training-point.schema.json',out,checkpoint=data.commit)
 finally:data.commit();models.commit()
@app.function(image=image,cpu=2,memory=8192,timeout=180)
def test_reader():
 import sys
 sys.path.insert(0,'/project/scripts')
 from test_hidden_cache_cpu import run_tests
 return run_tests()
@app.local_entrypoint()
def main(inspect_only:bool=False,test_only:bool=False):
 if test_only:
  result=test_reader.remote();p=ROOT/'outputs/hidden-cache/reader-tests.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 elif inspect_only:
  value=inspect_run.remote();p=ROOT/'outputs/hidden-cache/wandb-runs.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps([{'id':r['id'],'state':r['state'],'name':r['name'],'url':r['url']} for r in value],indent=2))
 else:
  result=build_cache.remote();p=ROOT/'outputs/hidden-cache/remote-manifest.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'records':result['identity']['record_count'],'tokens':result['identity']['total_tokens'],'bytes':result['total_bytes']},indent=2))
