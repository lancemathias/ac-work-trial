"""Pinned official PPLX environment and public-data reconstruction, CPU only."""
import hashlib
import json
import subprocess
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/project')
ENV = 'worktrial-lance'
UPSTREAM = ROOT / 'vendor/pplx-decider-v1-27b'
app = modal.App('ac-worktrial-pplx-v1-replication')
# uv.lock is byte-identical to the lock recorded for the original training run.
image = (modal.Image.debian_slim(python_version='3.12').apt_install('git')
         .pip_install('uv==0.13.1', 'wandb==0.22.3')
         .add_local_dir(UPSTREAM, '/opt/autojev', copy=True)
         .run_commands('uv sync --project /opt/autojev --frozen --no-dev')
         .add_local_dir(ROOT / 'scripts', '/project/scripts')
         .add_local_dir(ROOT / 'configs', '/project/configs')
         .add_local_dir(ROOT / 'data/manifests', '/project/data/manifests')
         .add_local_dir(ROOT / 'data/processed', '/project/data/processed'))
data = modal.Volume.from_name('ac-worktrial-training-data', environment_name=ENV)
models = modal.Volume.from_name('ac-worktrial-models', environment_name=ENV)
results = modal.Volume.from_name('ac-worktrial-training-results', environment_name=ENV)


@app.function(image=image, cpu=8, memory=32768, timeout=8*3600, retries=0,
              volumes={'/data': data, '/models': models})
def reconstruct(source):
    import os
    root = Path('/data/pplx-v1-public-reconstruction')
    root.mkdir(exist_ok=True)
    receipt = root / 'request.json'
    if receipt.exists() and json.loads(receipt.read_text()) != source:
        raise ValueError('Different reconstruction request cannot reuse this directory')
    receipt.write_text(json.dumps(source, indent=2) + '\n')
    environment = dict(os.environ, AUTOJEV_EVENTS=str(root / 'events.jsonl'), PYTHONUNBUFFERED='1')
    command = ['/opt/autojev/.venv/bin/autojev-data', '--cache-root', '/models/hub',
               '--source-cache', '/data/pplx-v1-public-raw', '--output', str(root / 'data'),
               '--manifest', str(root / 'manifest.json'), '--seed', '20260920',
               '--max-length', '8192', '--train-scale', '1.0', '--image-train', '1000', '--image-eval', '200']
    try:
        with (root / 'build.log').open('ab') as stream:
            subprocess.run(command, cwd='/opt/autojev', env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        files = [p for p in root.rglob('*') if p.is_file() and p.name != 'checksums.json']
        hashes = {}
        for path in files:
            with path.open('rb') as stream:
                hashes[str(path.relative_to(root))] = hashlib.file_digest(stream, 'sha256').hexdigest()
        (root / 'checksums.json').write_text(json.dumps(hashes, indent=2) + '\n')
        return {'status': 'complete', 'directory': str(root), 'manifest': json.loads((root / 'manifest.json').read_text()),
                'exact_curated_reproduction': False, 'reason': 'The release explicitly omits its curated 73k corpus; this is its published earlier 96k reconstruction.'}
    finally:
        data.commit()
        models.commit()


@app.local_entrypoint()
def prepare():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    source = {'upstream': json.loads((UPSTREAM / 'UPSTREAM.json').read_text()),
              'recipe': json.loads((UPSTREAM / 'configs/training.json').read_text()),
              'scope': 'Published public-data reconstruction only; no GPU training is started by this job.'}
    call = reconstruct.spawn(source)
    receipt = {'function_call_id': call.object_id, 'environment': ENV, 'cpu_only': True,
               'dataset': 'pplx-v1-public-reconstruction'}
    (ROOT / 'outputs/autoresearch/pplx-reconstruction-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
