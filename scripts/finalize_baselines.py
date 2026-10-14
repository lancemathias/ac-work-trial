"""Finalize the four-model baseline receipts, findings, and artifact checksum index."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/baselines-20261014'
KEYS = ['pplx', 'clef-flash', 'clef', 'jev']

def finalize():
    metrics = json.loads((BASE / 'metrics.json').read_text())
    launch_path = BASE / 'modal-launches.json'
    launch = json.loads(launch_path.read_text())
    for run in launch['runs']:
        if run['status'] == 'running':
            run['status'] = 'complete'
    costs = {}
    for key in KEYS[:3]:
        receipt = json.loads((BASE / key / 'completion.json').read_text())
        if receipt['capped']:
            raise ValueError('Cannot finalize a capped run as complete')
        costs[key] = receipt['gpu_function_wall_seconds'] * .000694
    launch['gpu_cost_estimates'] = costs
    launch['completed_function_gpu_cost_estimate_total'] = sum(costs.values())
    launch_path.write_text(json.dumps(launch, indent=2) + '\n')

    text = ['# Sampled baselines — findings', '',
        'Completed on 2026-10-14. Four released decision models were evaluated on the same frozen 2,111-decision panel. No training or calibration fitting occurred. This independently defined sampled protocol is not a reproduction of Perplexity’s undisclosed panel.', '',
        '| Model | 11-suite macro accuracy | Valid predictions |', '|---|---:|---:|']
    for key, label in zip(KEYS, ['PPLX v1-27B', 'CLEF-flash', 'CLEF', 'Jev 1.13.0']):
        n = sum(metrics[key][s]['valid'] for s in metrics[key] if s != 'overall')
        overall = metrics[key]['overall']
        v, a = overall['suite_macro_accuracy'], overall['coverage_adjusted_suite_macro_accuracy']
        cell = f'{100*v:.2f}%' if n == 2111 else f'{100*v:.2f}% / {100*a:.2f}%'
        text.append(f'| {label} | {cell} | {n}/2111 |')
    text += ['',
        'Dual scores appear only for misses: exclude misses / count misses as wrong. BBH is averaged across its included subtasks; overall equally weights 11 benchmarks. Both CLEF variants rejected ContractNLI:58:nda-8 at the common 8,192-token ceiling. PPLX’s native prompt fit that input. Jev used its hosted context limits. No input was silently shortened.', '',
        'Jev has the highest observed macro accuracy on this panel. CLEF leads JevBench hard, PPLX leads TabFact, and CLEF-flash leads WinoGrande. These are descriptive point estimates, not significance claims. FinancialPhraseBank uses all-agree examples, Belebele is English-only, JudgeBench uses one ordering, and Circa is a source-wide sample rather than a canonical held-out split. All assumptions are in baselines.md.', '',
        'Calibration tables report NLL, multiclass sum Brier and 15-bin ECE on valid predictions. PPLX uses its released temperature 2.207568021892729; CLEF variants use released softmax; Jev returns hosted probabilities. Eight Jev predictions assign reported zero probability to the gold label. NLL includes a 1e-3 floor sensitivity check against the primary 1e-12 floor. No benchmark labels were used for recalibration. Small samples and correlated examples limit calibration conclusions.', '',
        '| GPU run | Function minutes including loading | GPU-only estimate |', '|---|---:|---:|']
    for key in KEYS[:3]:
        c = json.loads((BASE / key / 'completion.json').read_text())
        text.append(f'| {key} | {c["gpu_function_wall_seconds"]/60:.2f} | ${costs[key]:.2f} |')
    text += ['',
        f'Completed-function GPU-only estimate: ${sum(costs.values()):.2f}, at Modal’s published A100-80GB rate of $0.000694/second ([Modal pricing](https://modal.com/pricing)). This is not an invoice: CPU, memory, startup/idle, the aborted import-error launch and Jev API charges are excluded. All successful GPU runs finished below their one-hour caps. PPLX and CLEF-flash ran concurrently; CLEF ran afterward with cost approval.', '',
        'Local batch-one latency includes encoding and synchronization; Jev latency includes network and retries. Reference convolution/linear-attention kernels were used. These are not optimized-serving measurements.', '',
        'All four runner hashes match source in their recorded Git commits. Prediction IDs, labels and panel checksums were validated before scoring. Configs, traces, receipts, result tables and calibration metrics are in outputs/baselines-20261014. Dataset text stays out of Git; source links, immutable revisions and checksums are committed.']
    (ROOT / 'docs/baseline-results.md').write_text('\n'.join(text) + '\n')
    index = {'kind': 'baselines', 'finalizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'panel_sha256': hashlib.sha256((ROOT / 'data/manifests/baseline-panel.json').read_bytes()).hexdigest(), 'files': {}}
    files = [BASE / name for name in ['metrics.json', 'results.md', 'modal-launches.json', 'model-runtime-manifest.json']]
    files += [BASE / key / name for key in KEYS for name in ['config.json', 'predictions.jsonl', 'completion.json']]
    for p in files:
        index['files'][str(p.relative_to(BASE))] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
    (BASE / 'artifact-index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(f'Completed-function GPU-only estimate: ${sum(costs.values()):.2f}')

if __name__ == '__main__': finalize()
