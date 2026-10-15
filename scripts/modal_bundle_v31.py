"""CPU-only portable bundles for the completed v3.1 dataset and training run."""
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
from cached_v31 import DATASET

app = modal.App('ac-worktrial-v31-artifact-bundle')
RUN = 'readout1024-qwen38-v3.1-choice-permutation1-seed20261014'
BUNDLE = 'v31-cached-readout-artifacts-20261015'


@app.function(image=image, cpu=4, memory=4096, timeout=1800, retries=0,
              volumes={'/results': results}, secrets=[secret])
def bundle():
    from head_probe_io import seal, write_json
    from training_data import digest
    out = Path('/results') / BUNDLE
    out.mkdir(exist_ok=True)
    roots = [Path('/results') / DATASET, Path('/results') / RUN]
    for root in roots:
        if json.loads((root / 'result.json').read_text())['status'] != 'complete':
            raise ValueError('Only completed artifacts can be bundled')
        hashes = json.loads((root / 'checksums.json').read_text())
        for name, expected in hashes.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != expected:
                raise ValueError('Artifact checksum mismatch')
    manifest = {}
    for root in roots:
        files = [p for p in sorted(root.rglob('*')) if p.is_file()]
        for path in files:
            manifest[str(path.relative_to('/results'))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        target = out / (root.name + '.tar.gz')
        temp = target.with_suffix('.tmp')
        with tarfile.open(temp, 'w:gz', compresslevel=1) as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to('/results')), recursive=False)
        temp.replace(target)
        results.commit()
    write_json(out / 'artifact-manifest.json', manifest)
    run = roots[1]
    import wandb
    info = json.loads((run / 'wandb-run.json').read_text())
    wb = wandb.init(project='ac-worktrial-decisions', id=info['id'], resume='must')
    artifact = wandb.Artifact('v31-choice-permutation1-readout1024-selected-head', type='model', metadata={
        'checkpoint_sha256': digest(run / 'best-head.pt'), 'dataset_manifest_sha256': digest(roots[0] / 'manifest.json')})
    for name in ['best-head.pt', 'best.json', 'config.json', 'temperature.json']:
        artifact.add_file(str(run / name), name=name)
    wb.log_artifact(artifact)
    wb.finish()
    value = {'status': 'complete', 'wandb': info['url'], 'archives': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in out.glob('*.tar.gz')}}
    write_json(out / 'result.json', value)
    seal(out)
    results.commit()
    return value


@app.local_entrypoint()
def main():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    call = bundle.spawn()
    receipt = {'function_call_id': call.object_id, 'bundle': BUNDLE, 'cpu_only': True}
    (ROOT / 'outputs/autoresearch/v31-artifact-bundle-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
