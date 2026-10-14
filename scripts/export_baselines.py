"""Export each Modal artifact explicitly; refuse to overwrite different evidence."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def export(models, profile='ac-worktrial', environment='ac-worktrial'):
    scratch = ROOT / 'work/modal-export'
    scratch.mkdir(parents=True, exist_ok=True)
    for model in models:
        out = ROOT / 'outputs/baselines-20261014' / model
        out.mkdir(parents=True, exist_ok=True)
        for name in ['config.json', 'predictions.jsonl', 'completion.json']:
            temporary = scratch / f'{model}-{name}'
            subprocess.run([str(ROOT / '.venv/bin/modal'), 'volume', 'get', '--profile', profile,
                            '--env', environment, 'ac-worktrial-baselines',
                            f'baselines-20261014/{model}/{name}', str(temporary), '--force'], check=True)
            payload = temporary.read_bytes()
            destination = out / name
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError(f'Existing evidence differs; preserve both runs: {destination}')
            destination.write_bytes(payload)
            print(model, name, hashlib.sha256(payload).hexdigest())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['pplx', 'clef-flash', 'clef'], required=True)
    parser.add_argument('--profile', default='ac-worktrial')
    parser.add_argument('--environment', default='ac-worktrial')
    args = parser.parse_args()
    export(args.models, args.profile, args.environment)
