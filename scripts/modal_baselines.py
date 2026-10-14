"""One serial A100 worker. Stop CLEF if PPLX reaches its one-hour cap."""
from pathlib import Path
import json
import subprocess
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12')
         .env({'BASELINE_GIT_COMMIT': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
         .pip_install('torch==2.14.0', 'transformers==5.17.0', 'accelerate==1.15.0',
                      'huggingface-hub==1.31.0', 'pillow==12.3.0', 'safetensors==0.8.0')
         .add_local_dir(ROOT / 'scripts', '/project/scripts')
         .add_local_dir(ROOT / 'data/processed', '/project/data/processed')
         .add_local_dir(ROOT / 'data/manifests', '/project/data/manifests'))
app = modal.App('ac-worktrial-baselines')
models = modal.Volume.from_name('ac-worktrial-models', create_if_missing=True)
results = modal.Volume.from_name('ac-worktrial-baselines', create_if_missing=True)

@app.function(image=image, gpu='A100-80GB', timeout=3600, max_containers=1,
              volumes={'/models': models, '/results': results})
def evaluate(model):
    import sys
    import time
    sys.path.insert(0, '/project/scripts')
    from run_baselines import run_local
    started = time.monotonic()
    out = Path('/results/baselines-20261014') / model
    # Reserve time for finalization; inference budget includes model loading.
    try:
        run_local(model, out, cap_seconds=3500, checkpoint_callback=results.commit)
    finally:
        results.commit()
        models.commit()
    completion = json.loads((out / 'completion.json').read_text())
    completion['gpu_function_wall_seconds'] = time.monotonic() - started
    return completion

@app.local_entrypoint()
def main():
    pplx = evaluate.remote('pplx')
    print('PPLX', pplx)
    print('CLEF-flash', evaluate.remote('clef-flash'))
    if not pplx['capped']:
        print('CLEF', evaluate.remote('clef'))
    else:
        print('CLEF skipped: PPLX reached its cap')
