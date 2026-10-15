"""Replay frozen preparation and deterministic schema generation; fail on byte drift."""
import argparse, hashlib, json, os, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(script,*args):
 env=dict(os.environ);env['PYTHONHASHSEED']='0'
 subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,env=env,check=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--rebuild-native',action='store_true');a.add_argument('--evaluation-dir',type=pathlib.Path,default=pathlib.Path('/Users/bwt/ac-worktrial/data/processed'));args=a.parse_args()
 if args.rebuild_native:
  if sys.version_info[:3]!=(3,14,7):raise RuntimeError('Native sampler replay requires recorded Python 3.14.7; use frozen native sample otherwise.')
  run('fetch_training_raw.py')
  for script in ['prepare_training_nli_operational.py','prepare_training_reading.py','prepare_training_specialist.py']:run(script)
  expected=json.loads((ROOT/'configs/training-native-output-hashes.json').read_text())
  for path,h in expected.items():
   actual=sha(ROOT/path)
   if actual!=h:raise ValueError(f'Native replay drift: {path}: {actual} != {h}')
  run('finalize_training_sample.py','--evaluation-dir',args.evaluation_dir)
 run('generate_typed_training.py','--seed','20261016')
 run('verify_typed_training.py')
 subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*typed*.py'],cwd=ROOT,check=True)
 print('Replay complete; source and typed hashes verified.')
if __name__=='__main__':main()
