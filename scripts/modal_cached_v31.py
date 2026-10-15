"""Build v3.1 on CPU, then fit a small cached head; never load the backbone."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path('/project/scripts').is_dir():
    sys.path.insert(0, '/project/scripts')
import modal
from modal_residual_readout import ROOT, ENV, image, results, secret, DEADLINE

app = modal.App('ac-worktrial-v31-cached-readout')


def identity():
    names = ['scripts/cached_v31.py', 'scripts/modal_cached_v31.py', 'scripts/residual_code_readout.py',
             'scripts/train_residual_readout.py', 'scripts/finish_fixed_readout.py', 'scripts/run_baselines.py',
             'tests/test_cached_v31.py', 'tests/test_finish_fixed_readout.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0, volumes={'/results': results})
def prepare(source):
    p = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_cached_v31', 'test_finish_fixed_readout'],
                       cwd='/project/tests', capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-12000:])
    from cached_v31 import build
    try:
        value = build('/results', source, results.commit)
        return {'tests': p.stderr, 'dataset': value}
    finally:
        results.commit()


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=16384, timeout=1200,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={'/results': results}, secrets=[secret])
def fit(run_id, source):
    import time
    if time.time() + 1200 > DEADLINE:
        raise TimeoutError('Reservation ended')
    if source['sha256'] != identity():
        raise ValueError('Source differs from validated snapshot')
    from cached_v31 import train
    try:
        return train('/results', run_id, source, results.commit)
    finally:
        results.commit()


@app.local_entrypoint()
def main(run_id: str = 'readout1024-qwen38-v3.1-choice-permutation1-seed20261014'):
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'sha256': identity()}
    prepared = prepare.remote(source)
    (ROOT / 'outputs/autoresearch/v31-dataset-preparation.json').write_text(json.dumps(prepared, indent=2) + '\n')
    call = fit.spawn(run_id, source)
    receipt = {'run_id': run_id, 'function_call_id': call.object_id, 'source': source,
               'profile': 'ac-worktrial', 'environment': ENV, 'backbone_loaded': False}
    (ROOT / 'outputs/autoresearch/v31-readout-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
