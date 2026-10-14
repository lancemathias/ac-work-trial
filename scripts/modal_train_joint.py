"""E1 launcher. Always use --profile ac-worktrial and --env worktrial-lance."""
import json
import re
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = 'worktrial-lance'
if modal.is_local():
    from modal.config import config, _profile
    if _profile != 'ac-worktrial' or config.get('environment') != ENVIRONMENT:
        raise ValueError('Use modal run --profile ac-worktrial --env worktrial-lance')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(ROOT/'requirements-training.txt')
    .add_local_dir(ROOT/'scripts', '/project/scripts')
    .add_local_dir(ROOT/'configs', '/project/configs'))
app = modal.App('ac-worktrial-e1-training')
# Approved training data only. No benchmark/test Volume is mounted.
data = modal.Volume.from_name('ac-worktrial-training-data', environment_name=ENVIRONMENT)
models = modal.Volume.from_name('ac-worktrial-models', environment_name=ENVIRONMENT, create_if_missing=True)
results = modal.Volume.from_name('ac-worktrial-training-results', environment_name=ENVIRONMENT, create_if_missing=True)
wandb_secret = modal.Secret.from_name('ac-worktrial-wandb', environment_name=ENVIRONMENT,
                                      required_keys=['WANDB_API_KEY'])


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=32768, timeout=12*3600, max_containers=1,
              volumes={'/data': data, '/models': models, '/results': results},
              secrets=[wandb_secret])
def train(manifest, run_id):
    import sys
    sys.path.insert(0, '/project/scripts')
    from train_joint import run
    from training_data import digest
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid run ID')
    path = (Path('/data')/manifest).resolve()
    if not path.is_relative_to('/data'):
        raise ValueError('Manifest must be inside the training data Volume')
    cfg = json.loads(Path('/project/configs/e1-qwen35-9b.json').read_text())
    dataset = Path('/data')/cfg['dataset']['path'].lstrip('/')
    if digest(dataset) != cfg['dataset']['sha256']:
        raise ValueError('Pinned source dataset checksum mismatch')
    try:
        return run('/project/configs/e1-qwen35-9b.json', path,
                   Path('/results')/run_id, checkpoint_callback=results.commit)
    finally:
        results.commit()
        models.commit()


@app.local_entrypoint()
def main(manifest: str, run_id: str):
    # Guard against launching the app in a different profile/environment.
    from modal.config import config
    if config.get('environment') != ENVIRONMENT:
        raise ValueError('Launch with --env worktrial-lance')
    from modal.config import _profile
    if _profile != 'ac-worktrial':
        raise ValueError('Launch with --profile ac-worktrial (workspace ac-worktrials)')
    print(json.dumps(train.remote(manifest, run_id), indent=2))
