"""Matched E6 output-zero initialization ablation; frozen cached features only."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/project')
ENV = 'worktrial-lance'
RUN = 'e6-zero-output-init-v31-seed20261014'
app = modal.App('ac-worktrial-e6-zero-output-init')
results = modal.Volume.from_name('ac-worktrial-training-results', environment_name=ENV)
secret = modal.Secret.from_name('ac-worktrial-wandb', environment_name=ENV, required_keys=['WANDB_API_KEY'])
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(ROOT / 'requirements-training.txt')
    .add_local_dir(ROOT / 'scripts', '/project/scripts')
    .add_local_dir(ROOT / 'configs', '/project/configs')
    .add_local_dir(ROOT / 'data/manifests', '/project/data/manifests')
    .add_local_dir(ROOT / 'data/processed', '/project/data/processed'))


def identity():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'scripts').rglob('*.py'))}


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=16384,
              timeout=1800, retries=0, max_containers=1,
              volumes={'/results': results}, secrets=[secret])
def fit(source):
    sys.path.insert(0, '/project/scripts')
    import tarfile
    import torch
    from cached_v31 import train
    from residual_code_readout import ResidualReadout
    if identity() != source['sha256']:
        raise ValueError('Uploaded source mismatch')
    # Verify zero output preserves the base and that both layers can learn.
    torch.manual_seed(123)
    probe = ResidualReadout(torch.randn(3, 8), width=4)
    torch.nn.init.zeros_(probe.down.weight)
    torch.nn.init.zeros_(probe.down.bias)
    x, counts = torch.randn(6, 8), torch.full((6,), 3)
    opt = torch.optim.SGD(probe.parameters(), lr=.1)
    torch.testing.assert_close(probe(x, counts), x @ probe.code_weight.T, rtol=0, atol=0)
    for step in range(2):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(probe(x, counts), torch.arange(6) % 3).backward()
        assert probe.down.weight.grad.abs().sum() > 0
        assert (probe.up.weight.grad.abs().sum() == 0) if step == 0 else (probe.up.weight.grad.abs().sum() > 0)
        opt.step()
    try:
        value = train('/results', RUN, source, results.commit, zero_init_output=True)
        out = Path('/results') / RUN
        (out / 'trainability-check.json').write_text(json.dumps({'base_parity': True, 'output_gradient_step1': True, 'hidden_gradient_step2': True}))
        from head_probe_io import seal
        seal(out)
        archive = Path('/results') / (RUN + '.tar.gz')
        with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
            bundle.add(out, arcname=RUN)
        return {'result': value, 'archive': archive.name,
                'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
    finally:
        results.commit()


@app.local_entrypoint()
def main():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'sha256': identity()}
    call = fit.spawn(source)
    receipt = {'run_id': RUN, 'function_call_id': call.object_id, 'source': source,
               'profile': 'ac-worktrial', 'workspace': 'ac-worktrials', 'environment': ENV}
    output = ROOT / 'outputs/autoresearch'
    (output / (RUN + '-launch.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'function_call_id': call.object_id, 'run_id': RUN}), flush=True)
    value = call.get()
    (output / (RUN + '-result.json')).write_text(json.dumps(value, indent=2) + '\n')
    target = ROOT / 'outputs/model-artifacts' / value['archive']
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as stream:
        for chunk in results.read_file(value['archive']):
            stream.write(chunk)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == value['sha256']
    (output / (RUN + '-export.json')).write_text(json.dumps({'path': str(target), 'sha256': value['sha256'], 'bytes': target.stat().st_size}, indent=2) + '\n')
    print(json.dumps(value, indent=2))
