"""One A100, one frozen backbone forward per example, three untrained heads."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.14.0','torchvision==0.29.0','transformers==5.17.0',
                 'accelerate==1.15.0','huggingface-hub==1.31.0','pillow==12.3.0','safetensors==0.8.0')
    .add_local_dir(ROOT/'scripts','/project/scripts')
    .add_local_dir(ROOT/'data/processed','/project/data/processed')
    .add_local_dir(ROOT/'data/manifests','/project/data/manifests'))
app=modal.App('ac-worktrial-random-joint')
models=modal.Volume.from_name('ac-worktrial-models',create_if_missing=True)
results=modal.Volume.from_name('ac-worktrial-baselines',create_if_missing=True)

@app.function(image=image,gpu='A100-80GB',timeout=3600,max_containers=1,
    volumes={'/models':models,'/results':results})
def run(variant: str = "qwen38-27b"):
    import sys
    sys.path.insert(0,'/project/scripts')
    from random_joint_baseline import evaluate, RUN, RUN_9B, BACKBONE, BACKBONE_9B, HEAD_CONFIG
    if variant not in ("qwen38-27b", "qwen35-9b"):
        raise ValueError("Unknown variant")
    small = variant == "qwen35-9b"
    run_name = RUN_9B if small else RUN
    head_config = dict(HEAD_CONFIG, hidden_size=4096 if small else 5120)
    try:
        return evaluate(Path('/results/baselines-20261014')/run_name, checkpoint_callback=results.commit,
            backbone_spec=BACKBONE_9B if small else BACKBONE, head_config=head_config)
    finally:
        results.commit()
        models.commit()

@app.local_entrypoint()
def main(variant: str = "qwen38-27b"):
    print(run.remote(variant))
