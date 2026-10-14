"""Export an entire run from Modal, verify checksums, keep local durable evidence."""
import argparse
import json
import tempfile
from pathlib import Path
from training_data import digest, validate_run_id


def download_file(remote, local):
    # Read explicit regular files; unlike CLI get, unsupported entries raise.
    import modal
    from modal.config import _set_profile
    _set_profile('ac-worktrial')
    volume = modal.Volume.from_name('ac-worktrial-training-results', environment_name='worktrial-lance')
    with Path(local).open('wb') as stream:
        for chunk in volume.read_file(remote):
            stream.write(chunk)


def export(run_id, destination):
    validate_run_id(run_id)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a fresh local destination; preserve existing evidence')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        target = Path(tmp)/'run'
        target.mkdir()
        download_file(f'{run_id}/checksums.json', target/'checksums.json')
        checksums = json.loads((target/'checksums.json').read_text())
        if not isinstance(checksums, dict) or not checksums:
            raise ValueError('Empty or invalid export manifest')
        for name, expected in checksums.items():
            path = (target/name).resolve()
            if not path.is_relative_to(target.resolve()) or path == target.resolve() or name == 'checksums.json':
                raise ValueError(f'Export path escapes evidence directory: {name}')
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f'Invalid checksum: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            download_file(f'{run_id}/{name}', path)
            if digest(path) != expected:
                raise ValueError(f'Export checksum mismatch: {name}')
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
