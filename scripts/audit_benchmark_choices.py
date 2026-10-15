"""Count candidate choices in the pinned local normalized benchmark corpora."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def stats(hist):
    values = sorted(k for k, n in hist.items() for _ in range(n))
    if not values:
        return None
    return {'min': min(values), 'max': max(values), 'mean': sum(values)/len(values),
            'median': (values[(len(values)-1)//2]+values[len(values)//2])/2,
            'p90_nearest_rank': values[math.ceil(.9*len(values))-1],
            'p95_nearest_rank': values[math.ceil(.95*len(values))-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('/Users/bwt/ac-worktrial/data/processed'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/benchmark-choice-audit.json')
    args = parser.parse_args()
    total = collections.Counter()
    results = []
    violations = []
    for path in sorted(args.directory.glob('*.jsonl')):
        histogram = collections.Counter()
        types = collections.Counter()
        for line in path.open():
            row = json.loads(line)
            question = row['question']
            types[question['type']] += 1
            if question['type'] != 'choice':
                continue
            options = question['criteria']
            histogram[len(options)] += 1
            if len(options) < 2 or row['label'] not in options:
                violations.append({'benchmark': path.stem, 'id': row['id'], 'reason': 'fewer than two choices or invalid gold membership'})
        total.update(histogram)
        results.append({'benchmark': path.stem, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'question_types': dict(types), 'choice_questions': sum(histogram.values()),
                        'option_count_histogram': dict(sorted(histogram.items())), 'statistics': stats(histogram)})
    report = {'scope': 'All records in the 11 local normalized benchmark corpora, not verified exact Perplexity subsets or per-run sampled baseline rows.',
              'counting_rule': 'Number of keys in question.criteria for question.type == choice; boolean and score questions excluded.',
              'choice_questions': sum(total.values()), 'option_count_histogram': dict(sorted(total.items())),
              'statistics': stats(total), 'by_benchmark': results, 'structural_violations': violations,
              'limitations': 'Option IDs are unique by dictionary representation. This audit checks gold membership and minimum count; it does not establish semantic uniqueness, label correctness, or detect duplicate JSON keys discarded during parsing.',
              'randomness': 'None; exhaustive counts.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
