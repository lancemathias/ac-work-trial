"""Deterministic grouped E1 splits after an externally approved hygiene audit."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from jsonschema import Draft202012Validator
from training_data import adapt_training_point, canonical, digest, load_splits, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def assign_roles(points, seed):
    """Keep families and identical states together, including linked families."""
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    states = {}
    for point in points:
        family = point['family_id']
        state = canonical(json.loads(point['model_payload']['input'])['state'])
        if state in states:
            a, b = find(family), find(states[state])
            parents[max(a,b)] = min(a,b)
        else:
            find(family)
            states[state] = family
    roles = {}
    for point in points:
        group = find(point['family_id'])
        fraction = int(hashlib.sha256(canonical([seed, group]).encode()).hexdigest(), 16) / 2**256
        roles[point['id']] = 'train' if fraction < .8 else 'dev' if fraction < .9 else 'calibration'
    return roles


def prepare(dataset, expected_sha256, hygiene_report, destination, seed=20261014):
    dataset, report, destination = Path(dataset), Path(hygiene_report), Path(destination)
    if digest(dataset) != expected_sha256:
        raise ValueError('Source dataset checksum mismatch')
    audit = json.loads(report.read_text())
    if audit.get('approved') is not True or audit.get('dataset_sha256') != expected_sha256:
        raise ValueError('Hygiene report must approve this exact dataset checksum')
    points = read_jsonl(dataset)
    validator = Draft202012Validator(json.loads((ROOT/'configs/training-point.schema.json').read_text()))
    for point in points:
        adapt_training_point(point, validator)
    if len({p['id'] for p in points}) != len(points):
        raise ValueError('Duplicate point IDs')
    roles = assign_roles(points, seed)
    partitions = {role: [p for p in points if roles[p['id']] == role] for role in ('train','dev','calibration')}
    if not all(partitions.values()):
        raise ValueError('Grouped split produced an empty role; select a preregistered alternate seed')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(report, destination/'hygiene-report.json')
    manifest = dict(version=1, source_dataset_sha256=expected_sha256,
        training_point_schema_sha256=digest(ROOT/'configs/training-point.schema.json'),
        split_seed=seed, split_rule='SHA256([seed, connected family/state group]); 80/10/10 hash thresholds',
        hygiene=dict(approved=True, report=dict(path='hygiene-report.json', sha256=digest(destination/'hygiene-report.json'))), splits={})
    for role, rows in partitions.items():
        path = destination/f'{role}.jsonl'
        path.write_text(''.join(canonical(row)+'\n' for row in rows))
        manifest['splits'][role] = dict(role=role, evaluation_only=False,
            allowed_sources=sorted({row['source'] for row in rows}), count=len(rows),
            points=dict(path=path.name, sha256=digest(path)))
    manifest_path = destination/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    load_splits(manifest_path)
    return manifest_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--hygiene-report', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    print(prepare(args.dataset, args.expected_sha256, args.hygiene_report, args.destination, args.seed))
