"""Qwen3.5-9B corrected benchmark-format cache worker."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENVIRONMENT='worktrial-lance'
if modal.is_local():
 from modal.config import config,_profile
 if _profile!='ac-worktrial' or config.get('environment')!=ENVIRONMENT:raise ValueError('Use --profile ac-worktrial --env worktrial-lance')
app=modal.App('ac-worktrial-v2-batch4-comparison')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-hidden-cache.txt').add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs'))
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENVIRONMENT)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENVIRONMENT)
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3*3600,max_containers=1,volumes={'/data':data,'/models':models})
def compare():
 import sys
 sys.path.insert(0,'/project/scripts')
 from compare_cache_batch4 import run
 snapshot=json.loads(Path('/project/configs/hidden-cache-qwen35-9b-dataset-v2.json').read_text())
 try:
  result=run(snapshot,'/data/hidden-cache/qwen35-9b-dataset-v2-338a3584e896','/data/hidden-cache-comparisons/v2-qwen35-9b-batch4')
  return json.loads(json.dumps(result))
 finally:data.commit();models.commit()
@app.local_entrypoint()
def main():
 result=compare.remote();p=ROOT/'outputs/hidden-cache-qwen35-9b-dataset-v2/batch4-summary.json';p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
