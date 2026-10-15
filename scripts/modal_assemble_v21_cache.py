"""Qwen3.5-9B corrected benchmark-format cache worker."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENVIRONMENT='worktrial-lance'
if modal.is_local():
 from modal.config import config,_profile
 if _profile!='ac-worktrial' or config.get('environment')!=ENVIRONMENT:raise ValueError('Use --profile ac-worktrial --env worktrial-lance')
app=modal.App('ac-worktrial-v21-cache-assembly')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-hidden-cache.txt').add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs'))
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENVIRONMENT)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENVIRONMENT)
@app.function(image=image,cpu=8,memory=32768,timeout=3*3600,max_containers=1,volumes={'/data':data})
def assemble():
 import sys
 sys.path.insert(0,'/project/scripts')
 from assemble_v21_cache import run
 return run(data)
@app.local_entrypoint()
def main():
 result=assemble.remote();p=ROOT/'outputs/hidden-cache-qwen35-9b-v21-batch4/assembly.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
