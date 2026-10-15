"""Run mask behavior and existing training regression tests on pinned Modal CPU."""
import json,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
app=modal.App('ac-worktrial-attention-ablation-tests')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs').add_local_dir(ROOT/'tests','/project/tests'))
@app.function(image=image,cpu=4,memory=8192,timeout=600)
def tests():
 import sys
 p=subprocess.run([sys.executable,'-m','unittest','-v','test_backbone_attention','test_training','test_cached_training','test_training_runtime','test_adamw_groups','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=tests.remote();(ROOT/'outputs/autoresearch/attention-tests.json').write_text(json.dumps(result,indent=2)+'\n')
 print(result['stderr'][-8000:]);print('returncode',result['returncode'])
 if result['returncode']:raise RuntimeError('Attention tests failed; GPU ablation must not launch')
