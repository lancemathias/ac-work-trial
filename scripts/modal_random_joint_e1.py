"""Three-seed random-head control matching E1's checkpoint and inference precision."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
ENV = 'worktrial-lance'
if modal.is_local():
    from modal.config import config, _profile
    if _profile != 'ac-worktrial' or config.get('environment') != ENV:
        raise ValueError('Use --profile ac-worktrial --env worktrial-lance')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.9.1', 'torchvision==0.24.1', 'transformers==5.17.0',
                 'accelerate==1.15.0', 'huggingface-hub==1.31.0', 'pillow==12.3.0', 'safetensors==0.8.0')
    .add_local_dir(ROOT/'scripts', '/project/scripts')
    .add_local_dir(ROOT/'data/processed', '/project/data/processed')
    .add_local_dir(ROOT/'data/manifests', '/project/data/manifests'))
app = modal.App('ac-worktrial-random-joint-e1-control')
models = modal.Volume.from_name('ac-worktrial-models', environment_name=ENV)
results = modal.Volume.from_name('ac-worktrial-baselines', environment_name=ENV)
e1 = modal.Volume.from_name('ac-worktrial-training-results', environment_name=ENV)


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=32768, timeout=3600, max_containers=1,
              volumes={'/models': models, '/results': results, '/e1': e1.read_only()})
def run():
    import sys
    sys.path.insert(0, '/project/scripts')
    from random_joint_baseline import evaluate, RUN_9B, BACKBONE_9B, HEAD_CONFIG
    try:
        return evaluate(Path('/results/baselines-20261014')/RUN_9B,
            checkpoint_callback=results.commit, backbone_spec=BACKBONE_9B,
            head_config=dict(HEAD_CONFIG, hidden_size=4096, dropout=0.0),
            fp32_head_autocast=True,
            e1_initial_head='/e1/e1-qwen35-9b-seed20261014-6f89ea4/initial-head.safetensors',
            e1_initial_sha256='2fe908f8ac0f25f81238b4f4c59b23377dc27665532741483b043a1b99bcf957')
    finally:
        results.commit()
        models.commit()


@app.local_entrypoint()
def main():
    print(run.remote())
