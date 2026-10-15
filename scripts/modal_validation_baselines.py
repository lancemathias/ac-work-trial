"""Pinned CLEF-flash on E2 validation; authorized Modal environment only."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'data/manifests','/project/data/manifests'))
app=modal.App('ac-worktrial-e2-validation-clef-flash')
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=3600,max_containers=1,volumes={'/models':models,'/results':results})
def evaluate():
 import sys,json
 sys.path.insert(0,'/project/scripts')
 from validation_baselines import load_validation
 from run_baselines import run_local,write_json,sha
 splits=Path('/results/e2-qwen35-9b-v2-cache-seed20261014-18359ca-splits');rows,metadata=load_validation(splits)
 out=Path('/results/e2-validation-clef-flash')
 try:
  run_local('clef-flash',out,rows=rows,metadata=metadata,checkpoint_callback=results.commit)
 finally:
  if out.exists():write_json(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='checksums.json'})
  results.commit();models.commit()
@app.local_entrypoint()
def main():
 from modal.config import config,_profile
 if _profile!='ac-worktrial' or config.get('environment')!=ENV:raise ValueError('Use authorized Modal profile/environment')
 evaluate.remote()
