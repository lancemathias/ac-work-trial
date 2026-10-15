"""Derive the active source-filtered set without changing the frozen 5k release."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_SOURCES = {'HelpSteer2', 'BANKING77'}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source = ROOT / 'data/training/typed-5000.jsonl'
    target = ROOT / 'data/training/typed-4500-no-helpsteer2-banking77.jsonl'
    lines = source.read_bytes().splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line)['source'] not in EXCLUDED_SOURCES]
    target.write_bytes(b''.join(kept))
    rows = [json.loads(line) for line in kept]
    histogram = collections.Counter()
    by_source = {}
    types = collections.Counter()
    violations = []
    for row in rows:
        types.update(row['field_types'].values())
        for field, kind in row['field_types'].items():
            if kind != 'choice':
                continue
            prop = row['model_payload']['schema']['properties'][field]
            options = prop['enum']
            n = len(options)
            histogram[n] += 1
            by_source.setdefault(row['source'], collections.Counter())[n] += 1
            if n < 2 or len(set(options)) != n or row['gold'][field] not in options:
                violations.append({'id': row['id'], 'field': field})
    counts = sorted(n for n, freq in histogram.items() for _ in range(freq))
    def percentile(p):
        # Nearest-rank percentile, with rank >= 1.
        import math
        return counts[max(0, math.ceil(p * len(counts)) - 1)]
    report = {
        'input': str(source.relative_to(ROOT)), 'input_sha256': digest(source),
        'output': str(target.relative_to(ROOT)), 'output_sha256': digest(target),
        'excluded_sources': sorted(EXCLUDED_SOURCES), 'excluded_records': len(lines)-len(kept),
        'records': len(rows), 'questions_by_type': dict(types),
        'randomness': 'None; source exclusion preserves original bytes and row order.',
        'choice_questions': len(counts), 'option_count_histogram': dict(sorted(histogram.items())),
        'statistics': {'min': min(counts), 'max': max(counts), 'mean': sum(counts)/len(counts),
                       'median': (counts[(len(counts)-1)//2]+counts[len(counts)//2])/2,
                       'p90_nearest_rank': percentile(.9), 'p95_nearest_rank': percentile(.95)},
        'by_source': {s: {'questions': sum(c.values()), 'option_count_histogram': dict(sorted(c.items()))}
                      for s,c in sorted(by_source.items())},
        'structural_violations': violations,
        'limitations': 'Checks enum count, unique option IDs, and gold membership; does not establish semantic uniqueness or label correctness.'}
    out = ROOT / 'outputs/choice-question-audit.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    config = {'dataset_path': str(target.relative_to(ROOT)), 'dataset_sha256': digest(target),
              'records': len(rows), 'excluded_sources': sorted(EXCLUDED_SOURCES),
              'derivation_script': 'scripts/filter_training_and_audit_choices.py',
              'audit': str(out.relative_to(ROOT)),
              'note': 'Active dataset for future runs. Frozen 5000-record release and existing runs/caches retain their original identity.'}
    (ROOT/'configs/active-training-dataset.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
