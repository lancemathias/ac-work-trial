"""Parallel PPLX/Flash baselines; CLEF is a separate cost-gated invocation."""
from pathlib import Path
import json
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/project')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() if modal.is_local() else os.environ['BASELINE_GIT_COMMIT']
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.14.0', 'torchvision==0.29.0', 'transformers==5.17.0', 'accelerate==1.15.0',
                      'huggingface-hub==1.31.0', 'pillow==12.3.0', 'safetensors==0.8.0')
         .env({'BASELINE_GIT_COMMIT': commit}))
if modal.is_local():
    image = (image.add_local_dir(ROOT / 'scripts', '/project/scripts')
         .add_local_dir(ROOT / 'data/processed', '/project/data/processed')
         .add_local_dir(ROOT / 'data/manifests', '/project/data/manifests'))
app = modal.App('ac-worktrial-baselines')
models = modal.Volume.from_name('ac-worktrial-models', create_if_missing=True)
results = modal.Volume.from_name('ac-worktrial-baselines', create_if_missing=True)

@app.function(image=image, gpu='A100-80GB', timeout=3600, max_containers=2,
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
    from run_baselines import write_json
    write_json(out / 'completion.json', completion)
    results.commit()
    return completion

@app.local_entrypoint()
def main(model_key: str = ''):
    keys = [model_key] if model_key else ['pplx', 'clef-flash']
    if any(k not in ['pplx', 'clef-flash', 'clef'] for k in keys):
        raise ValueError('Unknown model')
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(evaluate.remote, key): key for key in keys}
        for future in as_completed(pending):
            key = pending[future]
            try:
                completion = future.result()
            except modal.exception.FunctionTimeoutError:
                completion = {'capped': True, 'status': 'hard_timeout', 'limit_seconds': 3600}
            except Exception as exc:
                completion = {'status': 'failed', 'error': type(exc).__name__}
            print(key, completion, flush=True)
    if not model_key:
        print('CLEF awaits GPU cost review; launch separately with --model-key clef.', flush=True)
