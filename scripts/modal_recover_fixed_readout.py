"""CPU-only report repair, W&B completion, and portable artifact bundles."""
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path('/project/scripts').is_dir():
    sys.path.insert(0, '/project/scripts')
import modal
from modal_residual_readout import ROOT, ENV, image, results, secret

app = modal.App('ac-worktrial-fixed-readout-cpu-recovery')
RUN = 'fixed-readout1024-qwen38-v3-cached-finish-wb-resume'
PARENT = 'residual-readout1024-qwen38-v3-order-seed20261014'
BUNDLE = 'fixed-readout-cpu-recovery-20261015'


@app.function(image=image, cpu=4, memory=8192, timeout=3600, retries=0,
              volumes={'/results': results}, secrets=[secret])
def recover(source):
    import time
    from head_probe_io import write_json, seal, prepare_run
    from training_data import digest
    from report_trained_joint import report
    from finish_fixed_readout import verified_cache
    # The regression and existing head tests execute without any GPU allocation.
    check = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_finish_fixed_readout'],
                           cwd='/project/tests', capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(check.stderr[-10000:])
    root = Path('/results') / RUN
    parent = Path('/results') / PARENT
    bundle = Path('/results') / BUNDLE
    request = {'source': source, 'run_id': RUN, 'parent': PARENT, 'original_error': "KeyError: 'latency_ms'"}
    prior = prepare_run(bundle, request)
    if prior:
        return prior
    write_json(bundle / 'cpu-tests.json', {'status': 'passed', 'tests': check.stderr})
    # Verify every cached tensor, then use only already saved predictions for reporting.
    inventories = {}
    for owner in [parent, root]:
        for cache in sorted((owner / 'features').glob('*')):
            if (cache / 'result.json').exists():
                cached = verified_cache(cache)
                inventories[str(cache.relative_to(Path('/results')))] = {'records': len(cached[1]), 'checksums_sha256': digest(cache / 'checksums.json')}
                del cached
    if json.loads((root / 'training-complete.json').read_text())['status'] != 'complete':
        raise ValueError('Training not complete')
    before = {str(p.relative_to(root)): digest(p) for p in [root / 'best-head.pt', root / 'epoch-5.pt',
             root / 'temperature.json', root / 'test-native/logits.jsonl',
             root / 'test-native/raw-predictions.jsonl', root / 'test-native/calibrated-predictions.jsonl']}
    summary = report(root / 'test-native', label='Fixed residual MLP v3; CPU-recovered report')
    if any(digest(root / name) != expected for name, expected in before.items()):
        raise ValueError('Recovery changed model, calibration, or predictions')
    write_json(root / 'recovery.json', {'original_error': request['original_error'], 'cpu_only': True,
                                       'training_repeated': False, 'inference_repeated': False,
                                       'unchanged_artifact_sha256': before, 'source': source})
    # Resume the same requested W&B run; attach the saved selected model there.
    import wandb
    wbinfo = json.loads((root / 'wandb-run.json').read_text())
    wb = wandb.init(project='ac-worktrial-decisions', id=wbinfo['id'], resume='must')
    wb.summary['test'] = summary
    wb.summary['fixed/cpu_recovery'] = {'status': 'complete', 'original_error': request['original_error']}
    artifact = wandb.Artifact('fixed-readout1024-v3-selected-head', type='model', metadata={
        'run_id': RUN, 'checkpoint_sha256': digest(root / 'best-head.pt'),
        'best': json.loads((root / 'best.json').read_text())})
    for name in ['best-head.pt', 'best.json', 'config.json', 'temperature.json', 'training-complete.json']:
        artifact.add_file(str(root / name), name=name)
    wb.log_artifact(artifact)
    wb.finish()
    write_json(root / 'result.json', {'status': 'complete', 'recovered_on_cpu': True,
                                     'best': json.loads((root / 'best.json').read_text()),
                                     'test': summary, 'wandb': wbinfo['url']})
    seal(root)
    write_json(bundle / 'cache-inventory.json', inventories)
    write_json(bundle / 'result-summary.json', json.loads((root / 'result.json').read_text()))
    results.commit()
    # Bundle all committed parent and fixed-run files, including partial shuffled
    # features and optimizer states. Never delete or rewrite the source artifacts.
    roots = [parent, root, Path('/results/fixed-readout1024-qwen38-v3-cached-finish-seed20261014')]
    files = [p for owner in roots if owner.exists() for p in sorted(owner.rglob('*')) if p.is_file()]
    manifest = {str(p.relative_to('/results')): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in files}
    write_json(bundle / 'artifact-manifest.json', manifest)
    for category in ['checkpoints-and-results', 'frozen-features']:
        archive = bundle / (category + '.tar.gz')
        temp = archive.with_suffix('.tmp')
        with tarfile.open(temp, 'w:gz', compresslevel=1) as tar:
            for path in files:
                is_feature = 'features' in path.relative_to('/results').parts
                if is_feature == (category == 'frozen-features'):
                    tar.add(path, arcname=str(path.relative_to('/results')), recursive=False)
        temp.replace(archive)
        results.commit()
        print(json.dumps({'event': 'cpu_recovery_archive', 'name': archive.name, 'bytes': archive.stat().st_size}), flush=True)
    result = {'status': 'complete', 'request': request, 'summary': summary, 'wandb': wbinfo['url'],
              'archives': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in bundle.glob('*.tar.gz')},
              'cpu_only': True, 'gpu_runs_started': 0}
    write_json(bundle / 'result.json', result)
    seal(bundle)
    results.commit()
    return result


@app.local_entrypoint()
def main():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    names = ['scripts/modal_recover_fixed_readout.py', 'scripts/run_baselines.py', 'tests/test_finish_fixed_readout.py']
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}}
    call = recover.spawn(source)
    receipt = {'function_call_id': call.object_id, 'bundle': BUNDLE, 'cpu_only': True, 'source': source}
    (ROOT / 'outputs/autoresearch/fixed-readout-cpu-recovery-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
