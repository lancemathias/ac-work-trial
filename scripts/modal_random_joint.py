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
def run():
    import sys
    sys.path.insert(0,'/project/scripts')
    from random_joint_baseline import evaluate,RUN
    try:
        return evaluate(Path('/results/baselines-20261014')/RUN,checkpoint_callback=results.commit)
    finally:
        results.commit()
        models.commit()

@app.local_entrypoint()
def main():
    print(run.remote())
