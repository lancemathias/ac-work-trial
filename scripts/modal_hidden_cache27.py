"""Independent Qwen3.8-27B cache worker, with explicit pinned 9B-run overrides."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENVIRONMENT='worktrial-lance'
if modal.is_local():
 from modal.config import config,_profile
 if _profile!='ac-worktrial' or config.get('environment')!=ENVIRONMENT:raise ValueError('Use --profile ac-worktrial --env worktrial-lance')
app=modal.App('ac-worktrial-hidden-cache-qwen38-27b')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-hidden-cache.txt').add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs'))
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENVIRONMENT)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENVIRONMENT)
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3*3600,max_containers=1,volumes={'/data':data,'/models':models})
def extract27():
 import sys
 sys.path.insert(0,'/project/scripts')
 from cache_hidden_runtime import extract
 snapshot=json.loads(Path('/project/configs/hidden-cache-qwen38-27b.json').read_text());cfg=snapshot['config']['config']
 out=Path('/data/hidden-cache/qwen38-27b-1d4bf0f2ff60-3b3d8db82da7')
 try:return extract(snapshot,Path('/data')/cfg['dataset']['path'].lstrip('/'),'/project/configs/training-point.schema.json',out,checkpoint=data.commit)
 finally:data.commit();models.commit()
@app.local_entrypoint()
def main():
 result=extract27.remote();p=ROOT/'outputs/hidden-cache-qwen38-27b/remote-manifest.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'records':result['identity']['record_count'],'bytes':result['total_bytes']}))
