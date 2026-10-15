"""One fixed-order continuation; no shuffled work and no watchdog."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path('/project/scripts').is_dir():
    sys.path.insert(0, '/project/scripts')
import modal
from modal_residual_readout import ROOT, ENV, image, models, results, secret, DEADLINE

app = modal.App('ac-worktrial-fixed-readout-cached')
PARENT = 'residual-readout1024-qwen38-v3-order-seed20261014'


def identity():
    names = ['scripts/finish_fixed_readout.py', 'scripts/modal_finish_fixed_readout.py',
             'scripts/train_residual_readout.py', 'scripts/residual_code_readout.py',
             'tests/test_finish_fixed_readout.py', 'configs/readout-mlp/codes-255.json']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


@app.function(image=image, cpu=4, memory=8192, timeout=300, retries=0)
def check():
    p = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_residual_readout', 'test_finish_fixed_readout'], cwd='/project/tests', capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-12000:])
    return {'status': 'passed', 'tests': p.stderr, 'source': identity()}


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=98304, timeout=5400,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={'/models': models.read_only(), '/results': results}, secrets=[secret])
def finish(run_id, source):
    import time
    if source['sha256'] != identity():
        raise ValueError('Validated source mismatch')
    if time.time() + 5400 > DEADLINE:
        raise TimeoutError('Insufficient reservation')
    from finish_fixed_readout import run
    try:
        return run(Path('/results') / PARENT, Path('/results') / run_id, source, results.commit, DEADLINE - 120)
    finally:
        results.commit()


@app.local_entrypoint()
def main(run_id: str = 'fixed-readout1024-qwen38-v3-cached-finish-wb-resume'):
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    validation = check.remote()
    (ROOT / 'outputs/autoresearch/fixed-readout-finish-cpu.json').write_text(json.dumps(validation, indent=2) + '\n')
    if validation['source'] != identity():
        raise ValueError('Source changed during validation')
    containers = json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal', 'container', 'list', '--profile', 'ac-worktrial', '--env', ENV, '--json'], text=True))
    others = [x for x in containers if x['app_name'] != 'ac-worktrial-fixed-readout-cached']
    if others:
        raise RuntimeError('Prior worker has not released')
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'sha256': identity()}
    call = finish.spawn(run_id, source)
    receipt = {'run_id': run_id, 'parent': PARENT, 'function_call_id': call.object_id,
               'source': source, 'profile': 'ac-worktrial', 'environment': ENV}
    (ROOT / 'outputs/autoresearch/fixed-readout-finish-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
