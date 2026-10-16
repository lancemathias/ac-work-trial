"""Wait for the current GPU to release, launch the requested pair, and export it."""
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODAL = '/Users/bwt/ac-worktrial/.venv/bin/modal'
ENV = 'worktrial-lance'
RUN = 'residual-readout1024-qwen38-v3-order-seed20261014'
DEADLINE = 1792125869


def status(phase, **extra):
    value = {'phase': phase, 'run_id': RUN, 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
    target = ROOT / 'outputs/autoresearch/residual-readout-host-status.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)
    print(json.dumps(value), flush=True)


def main():
    import modal
    from modal.config import _set_profile
    from export_training import export
    _set_profile('ac-worktrial')
    launch = ROOT / 'outputs/autoresearch/residual-readout-launch.json'
    if not launch.exists():
        while True:
            if time.time() + 8*3600 > DEADLINE:
                raise TimeoutError('Not enough original reservation remains; no GPU requested')
            containers = json.loads(subprocess.check_output([MODAL, 'container', 'list', '--profile', 'ac-worktrial', '--env', ENV, '--json'], text=True, timeout=30))
            if not containers:
                break
            status('waiting_for_existing_gpu_release', containers=containers)
            time.sleep(30)
        status('launching_requested_pair')
        subprocess.run([MODAL, 'run', '--detach', '--profile', 'ac-worktrial', '--env', ENV,
                        'scripts/modal_residual_readout.py::main', '--run-id', RUN], cwd=ROOT, check=True)
    receipt = json.loads(launch.read_text())
    if receipt['run_id'] != RUN or receipt['environment'] != ENV:
        raise ValueError('Unexpected launch receipt')
    call = modal.FunctionCall.from_id(receipt['function_call_id'])
    while True:
        try:
            result = call.get(timeout=45)
            break
        except TimeoutError:
            status('running_requested_pair', function_call_id=receipt['function_call_id'])
    if result.get('status') != 'complete':
        raise RuntimeError('Paired trial incomplete')
    (ROOT / 'outputs/autoresearch/residual-readout-result.json').write_text(json.dumps(result, indent=2) + '\n')
    destination = ROOT / 'outputs/model-artifacts' / RUN
    status('exporting_complete_results', destination=str(destination))
    if destination.exists():
        hashes = json.loads((destination / 'checksums.json').read_text())
        if not hashes or any(hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected for name, expected in hashes.items()):
            raise ValueError('Existing export did not verify')
    else:
        export(RUN, destination)
    summary = ROOT / 'outputs/autoresearch/residual-readout-export.json'
    summary.write_text(json.dumps({'status': 'verified', 'run_id': RUN, 'local_root': str(destination),
                                  'checksums_sha256': hashlib.sha256((destination / 'checksums.json').read_bytes()).hexdigest(),
                                  'persistent_modal_volumes_retained': True}, indent=2) + '\n')
    status('complete_export_verified', destination=str(destination))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, message=str(exc))
        raise
