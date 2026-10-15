"""PPLX recipe on a deterministic 10k corpus sample: full-weight 9B, two epochs."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import modal
sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path('/project/scripts').is_dir():
    sys.path.insert(0, '/project/scripts')
from modal_pplx_replication import ROOT, ENV, image, data, models, results

app = modal.App('ac-worktrial-pplx-9b-10k-2epoch')
RUN = 'pplx-recipe-qwen35-9b-10k-2epoch-seed20260920'
wandb_secret = modal.Secret.from_name('ac-worktrial-wandb', environment_name=ENV, required_keys=['WANDB_API_KEY'])
if modal.is_local():
    from dotenv import dotenv_values
    jev_value = dotenv_values(ROOT.parents[1] / '.env').get('JEV_API_KEY')
    if not jev_value:
        raise ValueError('Existing Jev credential required for the upstream selection reference')
    jev_secret = modal.Secret.from_dict({'JEV_API_KEY': jev_value})
else:
    jev_secret = modal.Secret.from_dict({})


def identity():
    names = ['scripts/prepare_pplx_9b.py', 'scripts/modal_pplx_9b.py', 'scripts/pplx_replication_runner.py',
             'configs/pplx-9b-10k-2epoch.json']
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={'/data': data, '/models': models}, secrets=[jev_secret])
def prepare(source):
    if source['sha256'] != identity():
        raise ValueError('Uploaded preparation source changed')
    try:
        run = subprocess.run(['/opt/autojev/.venv/bin/python', '/project/scripts/prepare_pplx_9b.py'],
                             cwd='/opt/autojev', check=True)
        manifest = json.loads(Path('/data/pplx-v1-10k-seed20260920/manifest.json').read_text())
        return {'status': 'complete', 'manifest': manifest}
    finally:
        data.commit()
        models.commit()


@app.function(image=image, gpu='A100-80GB', cpu=32, memory=327680, timeout=6*3600,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={'/data': data.read_only(), '/models': models.read_only(), '/results': results},
              secrets=[wandb_secret])
def train(source, preparation):
    import wandb
    if source['sha256'] != identity():
        raise ValueError('Uploaded training source changed')
    from head_probe_io import write_json
    cfg = json.loads((ROOT / 'configs/pplx-9b-10k-2epoch.json').read_text())
    dataset = Path('/data/pplx-v1-10k-seed20260920')
    current = json.loads((dataset / 'manifest.json').read_text())
    if current != preparation['manifest']:
        raise ValueError('Prepared dataset changed')
    from training_data import digest
    for name, sha in current['sha256'].items():
        if digest(dataset / name) != sha:
            raise ValueError('Input artifact changed: ' + name)
    root = Path('/results') / RUN
    root.mkdir(exist_ok=False)
    write_json(root / 'dataset-manifest.json', current)
    write_json(root / 'experiment.json', {'config': cfg, 'source': source})
    wb = wandb.init(project='ac-worktrial-decisions', name=RUN,
        config={'recipe': cfg, 'source': source, 'dataset': {k: v for k, v in current.items() if k != 'train_ids'}},
        job_type='pplx-recipe-full-weight-9b')
    wb.define_metric('optimizer_step')
    wb.define_metric('train/*', step_metric='optimizer_step')
    wb.define_metric('development/*', step_metric='optimizer_step')
    write_json(root / 'wandb-run.json', {'id': wb.id, 'url': wb.url})
    args = ['--train', str(dataset / 'train.jsonl'), '--temperature', str(dataset / 'temperature.jsonl'),
        '--development', str(dataset / 'development.jsonl'), '--reference', str(dataset / 'jev-reference.jsonl'),
        '--public', str(dataset / 'public.jsonl'), '--run', str(root / 'run'), '--output', str(root / 'checkpoints'),
        '--base-model', cfg['base_model'], '--revision', cfg['revision'], '--cache-dir', '/models/hub']
    for flag in ['epochs', 'seed', 'lr', 'weight_decay', 'batch_size', 'effective_batch_size', 'token_budget',
                 'max_length', 'cpu_threads', 'eval_every', 'public_eval_every', 'resume_every']:
        args += ['--' + flag.replace('_', '-'), str(cfg[flag])]
    spec = {'mode': 'full', 'root': str(root), 'argv': args, 'source': source,
        'deviations': ['Uniform 10,000-row sample of published reconstructed corpus instead of unavailable curated 73k corpus',
            'Qwen3.5-9B instead of Qwen3.8-27B', 'Two epochs instead of one',
            'Fresh Jev 1.13 reference on reconstructed development fold via existing typesafe.ai account',
            'W&B logging and post-training terminal/selected test evaluation; no other training algorithm changes']}
    write_json(root / 'spec.json', spec)
    results.commit()
    # Upstream records git provenance; preserve the exact source in a standalone git snapshot.
    subprocess.run(['git', 'init', '-q', '/opt/autojev'], check=True)
    subprocess.run(['git', '-C', '/opt/autojev', 'add', 'src', 'configs', 'pyproject.toml', 'uv.lock', 'README.md', 'LICENSE'], check=True)
    subprocess.run(['git', '-C', '/opt/autojev', '-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com',
                    'commit', '-q', '-m', 'Pinned upstream PPLX v1 source'], check=True)
    env = dict(os.environ, AUTOJEV_EVENTS=str(root / 'events.jsonl'), PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = ['/opt/autojev/.venv/bin/python', '/project/scripts/pplx_replication_runner.py', str(root / 'spec.json')]
    try:
        with (root / 'runtime.log').open('a') as log:
            process = subprocess.Popen(command, cwd='/opt/autojev', env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get('kind') == 'training_step':
                    wb.log({'optimizer_step': value['step'], **{'train/' + k: v for k, v in value.items()
                            if isinstance(v, (int, float)) and k != 'step'}})
                    results.commit()
                elif value.get('kind') == 'evaluation':
                    wb.log({'optimizer_step': value['step'], **{'development/' + k: v for k, v in value['fitted'].items()
                            if isinstance(v, (int, float))}, 'development/temperature': value['temperature']})
                    results.commit()
            if process.wait():
                raise RuntimeError('Training subprocess failed; see preserved runtime.log')
        value = json.loads((root / 'result.json').read_text())
        wb.summary['terminal_test'] = value['terminal']
        wb.summary['upstream_selected_test'] = value['upstream_selected']
        wb.summary['training_summary'] = value['training_summary']
        wb.finish()
        from head_probe_io import seal
        seal(root)
        return {'status': 'complete', 'run_id': RUN, 'wandb': wb.url, 'result': value}
    except Exception:
        wb.finish(exit_code=1)
        raise
    finally:
        results.commit()


@app.local_entrypoint()
def main():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'sha256': identity()}
    prepared = prepare.remote(source)
    (ROOT / 'outputs/autoresearch/pplx-9b-preparation.json').write_text(json.dumps(prepared, indent=2) + '\n')
    call = train.spawn(source, prepared)
    receipt = {'run_id': RUN, 'function_call_id': call.object_id, 'source': source, 'environment': ENV}
    (ROOT / 'outputs/autoresearch/pplx-9b-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
