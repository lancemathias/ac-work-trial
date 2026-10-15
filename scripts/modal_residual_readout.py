"""Experiment 5 residual MLP; original shuffled branch terminated and superseded by Experiment 6."""
import hashlib
import json
import subprocess
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/project')
ENV = 'worktrial-lance'
DEADLINE = 1792125869
SPLITS = 'outputs/model-artifacts/e3-medium-v3-seed20261014-e7dd4bb-splits'
app = modal.App('ac-worktrial-experiment-5-residual-mlp')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install_from_requirements(ROOT / 'requirements-training.txt')
         .pip_install_from_requirements(ROOT / 'requirements-kernels.txt', extra_options='--no-deps')
         .add_local_dir(ROOT / 'scripts', '/project/scripts')
         .add_local_dir(ROOT / 'configs', '/project/configs')
         .add_local_dir(ROOT / 'tests', '/project/tests')
         .add_local_dir(ROOT / SPLITS, '/project/' + SPLITS)
         .add_local_dir(ROOT / 'data/manifests', '/project/data/manifests')
         .add_local_dir(ROOT / 'data/processed', '/project/data/processed'))
models = modal.Volume.from_name('ac-worktrial-models', environment_name=ENV)
results = modal.Volume.from_name('ac-worktrial-training-results', environment_name=ENV)
secret = modal.Secret.from_name('ac-worktrial-wandb', environment_name=ENV, required_keys=['WANDB_API_KEY'])


def identity():
    names = ['scripts/residual_code_readout.py', 'scripts/train_residual_readout.py',
             'scripts/modal_residual_readout.py', 'configs/readout-mlp/experiment.json',
             'configs/readout-mlp/codes-255.json', 'tests/test_residual_readout.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={'/models': models.read_only()})
def check():
    import sys
    sys.path.insert(0, '/project/scripts')
    p = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_residual_readout'],
                       cwd='/project/tests', capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-15000:])
    from resident_backbone import get_processor
    from training_data import load_splits, digest
    from residual_code_readout import verify_codes, encode
    from run_baselines import load_panel
    cfg = json.loads((ROOT / 'configs/readout-mlp/experiment.json').read_text())
    mp = ROOT / cfg['split_manifest']
    if digest(mp) != cfg['split_manifest_sha256']:
        raise ValueError('v3 split manifest changed')
    splits, manifest = load_splits(mp)
    tokenizer = get_processor(cfg).tokenizer
    codes = verify_codes(tokenizer, cfg['backbone_id'])
    stats = {}
    for role, rows in splits.items():
        views = [('native', 0)] + ([('shuffle', e + 1) for e in range(cfg['epochs'])] if role == 'train' else [])
        for order, epoch in views:
            lengths = [len(encode(tokenizer, r, True, order, cfg['seed'], epoch)['input_ids']) for r in rows]
            stats[f'{role}-{order}-{epoch}'] = {'records': len(lengths), 'max_tokens': max(lengths), 'total_tokens': sum(lengths)}
    for order in cfg['evaluation_orders']:
        lengths = [len(encode(tokenizer, r, False, order, cfg['seed'], 1000000)['input_ids']) for r in load_panel()]
        stats['test-' + order] = {'records': len(lengths), 'max_tokens': max(lengths), 'total_tokens': sum(lengths)}
    return {'status': 'passed', 'tests': p.stderr, 'source_sha256': identity(),
            'codes': len(codes), 'dataset_sha256': manifest['source_dataset_sha256'], 'sets': stats}


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=98304, timeout=8*3600,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={'/models': models.read_only(), '/results': results}, secrets=[secret])
def train(run_id, source):
    import sys
    sys.path.insert(0, '/project/scripts')
    from train_residual_readout import run
    from head_probe_io import write_json
    from training_data import validate_run_id
    validate_run_id(run_id)
    if source['sha256'] != identity():
        raise ValueError('Validated source differs from uploaded runtime')
    if __import__('time').time() + 8*3600 > DEADLINE:
        raise TimeoutError('Insufficient original GPU reservation for this bounded pair')
    root = Path('/results') / run_id
    try:
        return run(root, ROOT / 'configs/readout-mlp/experiment.json', source, results.commit, DEADLINE - 120)
    except Exception as exc:
        root.mkdir(exist_ok=True, parents=True)
        write_json(root / 'failure.json', {'error': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        results.commit()


@app.local_entrypoint()
def validate():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    result = check.remote()
    target = ROOT / 'outputs/autoresearch/residual-readout-cpu.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def main(run_id: str = 'residual-readout1024-qwen38-v3-order-seed20261014'):
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    validation = json.loads((ROOT / 'outputs/autoresearch/residual-readout-cpu.json').read_text())
    if validation['status'] != 'passed' or validation['source_sha256'] != identity():
        raise ValueError('Fresh exact-source CPU validation required')
    containers = json.loads(subprocess.check_output([
        '/Users/bwt/ac-worktrial/.venv/bin/modal', 'container', 'list', '--profile', 'ac-worktrial', '--env', ENV, '--json'], text=True))
    if containers:
        raise RuntimeError('Wait for existing GPU work to release before launching this pair')
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'sha256': identity()}
    call = train.spawn(run_id, source)
    receipt = {'run_id': run_id, 'function_call_id': call.object_id, 'source': source,
               'profile': 'ac-worktrial', 'workspace': 'ac-worktrials', 'environment': ENV}
    (ROOT / 'outputs/autoresearch/residual-readout-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
