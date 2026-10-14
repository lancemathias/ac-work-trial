"""Export an entire run from Modal, verify checksums, keep local durable evidence."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from training_data import digest, validate_run_id


def export(run_id, destination):
    validate_run_id(run_id)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a fresh local destination; preserve existing evidence')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        target = Path(tmp)/'run'
        subprocess.run(['modal', 'volume', 'get', '--profile', 'ac-worktrial', '--env', 'worktrial-lance',
                        'ac-worktrial-training-results', run_id, str(target)], check=True)
        # Modal directory downloads may preserve the remote directory name.
        if not (target/'checksums.json').exists() and (target/run_id/'checksums.json').exists():
            target = target/run_id
        checksums = json.loads((target/'checksums.json').read_text())
        for name, expected in checksums.items():
            path = (target/name).resolve()
            if not path.is_relative_to(target.resolve()) or digest(path) != expected:
                raise ValueError(f'Export checksum/path mismatch: {name}')
        actual = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file() and p.name != 'checksums.json'}
        if actual != set(checksums):
            raise ValueError('Export file inventory mismatch')
        target.rename(destination)
    print(f'Export verified: {destination}; Modal resources retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    export(args.run_id, args.destination)
