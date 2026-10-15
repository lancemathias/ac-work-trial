"""Validate and publish E1-matched random-head metrics from exported traces."""
import json
from pathlib import Path
from random_joint_baseline import ROOT, RUN_9B, SEEDS, merge_table, summarize
from run_baselines import sha, write_json


def fmt(value):
    return f'{100*value["mean"]:.2f}% ± {100*value["sample_sd"]:.2f}'


def main():
    base = ROOT / 'outputs/baselines-20261014'
    folder = base / RUN_9B
    cfg = json.loads((folder/'config.json').read_text())
    assert cfg['runner_sha256'] == sha(folder/'runtime-runner.py')
    assert cfg['e1_initial_sha256'] == json.loads((folder/f'seed-{SEEDS[0]}/config.json').read_text())['initial_head_sha256']
    completion = json.loads((folder/'completion.json').read_text())
    assert completion['status'] == 'complete' and completion['backbone_parameters_unchanged']
    result = summarize(base, run=RUN_9B)
    assert all(x['attempted_per_seed'] == [x['expected']]*3 for x in result['suites'].values())
    merge_table(base, run=RUN_9B)
    lines = ['# E1-matched Qwen3.5-9B untrained-head baseline', '',
        'Three independent head initializations; mean ± sample standard deviation across seeds. '
        'Seed 20261014 uses E1’s saved pre-training head. The frozen backbone, architecture, '
        'FP32 head parameters and BF16 autocast match E1. T=1; no training or calibration fitting.', '',
        'Where coverage is incomplete, accuracy is valid-only / misses counted wrong. '
        'These are averages of scores, not an ensemble.', '',
        '| Benchmark | n per seed | Accuracy | Valid predictions per seed |',
        '|---|---:|---:|---|']
    for suite, row in result['suites'].items():
        value = fmt(row['accuracy'])
        if not row['complete']:
            value += ' / ' + fmt(row['coverage_adjusted_accuracy'])
        lines.append(f'| {suite} | {row["expected"]} | {value} | {row["valid_per_seed"]} |')
    value = fmt(result['valid_only_overall'])
    if result['overall'] is None:
        value += ' / ' + fmt(result['coverage_adjusted_overall'])
    lines += [f'| **11-suite macro** | **2111** | **{value}** | |', '',
        '| Seed | Valid-only macro | Misses counted wrong |', '|---|---:|---:|']
    for i, seed in enumerate(SEEDS):
        lines.append(f'| {seed} | {100*result["valid_only_overall"]["per_seed"][i]:.2f}% | '
                     f'{100*result["coverage_adjusted_overall"]["per_seed"][i]:.2f}% |')
    lines += ['', f'GPU worker elapsed: {completion["elapsed_seconds"]/60:.2f} minutes; '
              f'peak allocated memory: {completion["peak_allocated_bytes"]/2**30:.2f} GiB. '
              'All backbone parameter version counters remained unchanged.', '',
              'Full calibration metrics and seed scores are in metrics.json; checkpoint provenance '
              'and hashes are in config.json, e1-reference.json and initial-head-checksums.json.']
    (folder/'results.md').write_text('\n'.join(lines)+'\n')
    manifest = {'modal_app_id': 'ap-pApzwxBCWea1RomVgc4KCM',
                'files': {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
                          if p.is_file() and p.name != 'artifact-manifest.json'}}
    write_json(folder/'artifact-manifest.json', manifest)
    print(value)


if __name__ == '__main__':
    main()
