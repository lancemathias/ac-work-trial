"""Frozen decision-model baselines. Never fit or train on this evaluation panel."""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'pplx': ('perplexity-ai/pplx-decider-v1-27b', 'd8c3bd18e31dc1804433d38528b3e770019bf00c'),
    'clef-flash': ('Cloudflare/clef-flash', '8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3'),
    'clef': ('Cloudflare/clef', '0b331204bb13fbd2ca93a64df1956f5b55478ce5'),
    'jev': ('jev-1.13.0', None),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]

def freeze(limit=200, seed=20261014):
    destination = ROOT / 'data/manifests/baseline-panel.json'
    original = json.loads((ROOT / 'data/manifests/benchmark-preprocessing.json').read_text())
    suites = {}
    for name, spec in original['suites'].items():
        path = ROOT / 'data/processed' / (name + '.jsonl')
        if sha(path) != spec['sha256']:
            raise ValueError(f'Processed data changed: {name}')
        rows = read_jsonl(path)
        buckets = defaultdict(list)
        for row in rows:
            stratum = row['source'].get('subtask', 'all') if name == 'BBH' else 'all'
            buckets[stratum].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda r: hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest())
        selected = []
        while buckets and (limit == 0 or len(selected) < limit):
            for key in sorted(list(buckets)):
                if limit and len(selected) >= limit:
                    break
                selected.append(buckets[key].pop(0))
                if not buckets[key]:
                    del buckets[key]
        suites[name] = {'ids': [r['id'] for r in selected], 'count': len(selected),
                        'processed_sha256': sha(path), 'source_splits': sorted({r['source'].get('split', 'unspecified') for r in selected}),
                        'labels': dict(Counter(str(r['label']) for r in selected)),
                        'subtasks': dict(Counter(r['source'].get('subtask', 'all') for r in selected))}
    panel = {'kind': 'baselines', 'seed': seed, 'limit_per_suite': limit,
             'selection': 'Label-blind SHA256 ID ranking; round-robin BBH subtasks; retain all smaller suites.',
             'preprocessing_manifest_sha256': sha(ROOT / 'data/manifests/benchmark-preprocessing.json'),
             'models': MODELS, 'suites': suites}
    if destination.exists() and json.loads(destination.read_text()) != json.loads(json.dumps(panel)):
        raise ValueError('Existing frozen panel differs; use a new panel path/run rather than overwrite')
    write_json(destination, panel)
    print('Frozen', sum(s['count'] for s in suites.values()), 'decisions', sha(destination))

def load_panel():
    manifest = ROOT / 'data/manifests/baseline-panel.json'
    panel = json.loads(manifest.read_text())
    if sha(ROOT / 'data/manifests/benchmark-preprocessing.json') != panel['preprocessing_manifest_sha256']:
        raise ValueError('Preprocessing manifest changed')
    rows = []
    for suite, spec in panel['suites'].items():
        path = ROOT / 'data/processed' / (suite + '.jsonl')
        if sha(path) != spec['processed_sha256']:
            raise ValueError('Processed checksum mismatch: ' + suite)
        by_id = {r['id']: r for r in read_jsonl(path)}
        rows.extend(by_id[i] for i in spec['ids'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate panel ID')
    return rows

def payload(row, model):
    # Deliberately exclude label, provenance, group and every other evaluation field.
    return {'model': model, 'state': row['state'], 'questions': {'q': row['question']}}

def score(row, body):
    answer = body['answers']['q']
    question = row['question']
    if answer['type'] != question['type']:
        raise ValueError('Answer type mismatch')
    if question['type'] == 'noul':
        labels = [False, True]
        probabilities = [1 - float(answer['noul']), float(answer['noul'])]
    else:
        labels = list(question['criteria']) if question['type'] == 'choice' else list(range(len(question['criteria'])))
        distribution = answer['probabilities']
        if set(distribution) != {str(x) for x in labels}:
            raise ValueError('Answer options mismatch')
        probabilities = [float(distribution[str(x)]) for x in labels]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probability')
    total = sum(probabilities)
    if abs(total - 1) > .025:
        raise ValueError('Probability sum outside rounding tolerance')
    probabilities = [p / total for p in probabilities]
    index = max(range(len(labels)), key=probabilities.__getitem__)
    gold = next(i for i, x in enumerate(labels) if type(x) is type(row['label']) and x == row['label'])
    result = {'status': 'ok', 'prediction': labels[index], 'probabilities': probabilities,
              'options': labels, 'normalization_total': total, 'correct': index == gold,
              'confidence': probabilities[index], 'nll': -math.log(max(probabilities[gold], 1e-12)),
              'brier': sum((p - (i == gold)) ** 2 for i, p in enumerate(probabilities))}
    if question['type'] == 'score':
        result['score_mae'] = abs(sum(i * p for i, p in enumerate(probabilities)) - row['label'])
    return result

def config(model, cap_seconds):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        commit = os.environ.get('BASELINE_GIT_COMMIT')
    return {'kind': 'baselines', 'model': model, 'model_id': MODELS[model][0], 'revision': MODELS[model][1],
            'panel_sha256': sha(ROOT / 'data/manifests/baseline-panel.json'),
            'runner_sha256': sha(__file__), 'git_commit': commit, 'cap_seconds': cap_seconds,
            'prediction_rule': 'argmax for Choice/Noul/Score; first option breaks ties',
            'normalization': 'Normalize only sums within 0.025 of 1; retain original sum',
            'calibration': 'Released probabilities; no evaluation-set calibration',
            'created': datetime.now(timezone.utc).isoformat()}

def record(row, result):
    return {'id': row['id'], 'suite': row['suite'], 'group': row['group'],
            'subtask': row['source'].get('subtask'), 'label': row['label'], **result}

def completed(path):
    return {r['id'] for r in read_jsonl(path)} if path.exists() else set()

async def run_jev(out, cap_seconds=3600, concurrency=8):
    import httpx
    from dotenv import dotenv_values
    model = MODELS['jev'][0]
    key = dotenv_values(ROOT / '.env').get('JEV_API_KEY')
    if not key:
        raise RuntimeError('Missing JEV_API_KEY in .env')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg = config('jev', cap_seconds)
    if (out / 'config.json').exists():
        old = json.loads((out / 'config.json').read_text())
        for field in ['panel_sha256', 'runner_sha256', 'model_id']:
            if old[field] != cfg[field]:
                raise ValueError('Resume configuration mismatch: ' + field)
    else:
        write_json(out / 'config.json', cfg)
    path = out / 'predictions.jsonl'
    done = completed(path)
    queue = asyncio.Queue()
    for row in load_panel():
        if row['id'] not in done:
            queue.put_nowait(row)
    start = time.monotonic(); fatal = asyncio.Event(); counts = Counter()
    async with httpx.AsyncClient(timeout=90, headers={'Authorization': 'Bearer ' + key}) as client:
        async def worker():
            while not fatal.is_set() and time.monotonic() - start < cap_seconds and not queue.empty():
                row = queue.get_nowait(); request = payload(row, model); began = time.monotonic()
                result = None
                for attempt in range(3):
                    try:
                        response = await client.post('https://api.typesafe.ai/v1/systemone', json=request)
                        if response.status_code in (401, 403):
                            fatal.set(); result = {'status': 'error', 'error': f'HTTP {response.status_code}'}; break
                        if response.status_code == 429 or response.status_code >= 500:
                            if attempt < 2:
                                await asyncio.sleep(min(10, float(response.headers.get('retry-after', 2 ** attempt))))
                                continue
                        response.raise_for_status()
                        body = response.json()
                        if body.get('model') != model:
                            fatal.set(); raise ValueError('Returned model differs from pinned model')
                        result = {**score(row, body), 'response': body, 'attempts': attempt + 1}; break
                    except Exception as exc:
                        # Never stringify HTTP exceptions: they may include sensitive headers.
                        result = {'status': 'error', 'error': type(exc).__name__, 'attempts': attempt + 1}
                        if attempt < 2 and not fatal.is_set():
                            await asyncio.sleep(2 ** attempt)
                result.update({'latency_ms': 1000 * (time.monotonic() - began),
                               'request_sha256': hashlib.sha256(canonical(request).encode()).hexdigest()})
                with path.open('a') as stream:
                    stream.write(canonical(record(row, result)) + '\n'); stream.flush()
                counts[result['status']] += 1
                if sum(counts.values()) % 100 == 0:
                    print('Jev completed', sum(counts.values()), dict(counts), flush=True)
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    write_json(out / 'completion.json', {'elapsed_seconds': time.monotonic() - start,
               'remaining': queue.qsize(), 'fatal': fatal.is_set(), 'counts': dict(counts)})

def run_local(model_key, out, cap_seconds=3500, checkpoint_callback=None):
    import sys
    import torch
    from huggingface_hub import snapshot_download
    start = time.monotonic()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg = config(model_key, cap_seconds)
    cfg.update({'hardware': torch.cuda.get_device_name(), 'precision': 'bfloat16',
                'torch': torch.__version__, 'batch_size': 1, 'max_input_tokens': 8192})
    write_json(out / 'config.json', cfg)
    model_id, revision = MODELS[model_key]
    snapshot = snapshot_download(model_id, revision=revision, cache_dir='/models/hub')
    if model_key == 'pplx':
        sys.path.insert(0, str(Path(snapshot) / 'source/src'))
        from autojev.model import DecisionModel, answer
        engine = DecisionModel(checkpoint=snapshot, device='cuda').eval()
        def infer(row):
            data = {'state': row['state'], 'question': row['question']}
            batch = engine.prepare([data], max_length=8192)
            probs = (engine(batch)[0] / engine.temperature).softmax(-1).cpu().tolist()
            n = 2 if row['question']['type'] == 'noul' else len(row['question']['criteria'])
            return {'answers': {'q': answer(row['question'], probs[:n])}, 'usage': {'input_tokens': batch.input_tokens}}
    else:
        sys.path.insert(0, snapshot)
        from joint_schema_model import encode_record, collate_records, load_release_model, systemone_answer
        engine, processor = load_release_model(snapshot, device='cuda')
        def infer(row):
            request = payload(row, model_id)
            # Official encoder truncates silently. Encode unbounded, then explicitly reject.
            encoded = encode_record(processor.tokenizer, request, max_length=10_000_000, processor=processor)
            if len(encoded.input_ids) > 8192:
                raise ValueError('Input exceeds common 8192-token limit; no truncation')
            batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device('cuda'))
            logits = engine(batch)[0][0]
            p = logits.float().softmax(-1).cpu().tolist()
            a = systemone_answer(row['question'], dict(zip(encoded.questions[0].option_ids, p)))
            return {'answers': {'q': a}, 'usage': {'input_tokens': len(encoded.input_ids)}}
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - start
    path = out / 'predictions.jsonl'; done = completed(path); capped = False
    for row in load_panel():
        if row['id'] in done:
            continue
        if time.monotonic() - start >= cap_seconds:
            capped = True; break
        began = time.monotonic()
        try:
            with torch.inference_mode():
                body = infer(row)
            torch.cuda.synchronize()
            result = {**score(row, body), 'response': body}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); result = {'status': 'error', 'error': 'OutOfMemoryError'}
        except Exception as exc:
            result = {'status': 'error', 'error': type(exc).__name__, 'detail': str(exc)[:250]}
        result['latency_ms'] = 1000 * (time.monotonic() - began)
        with path.open('a') as stream:
            stream.write(canonical(record(row, result)) + '\n'); stream.flush()
        if len(completed(path)) % 100 == 0:
            print(model_key, 'completed', len(completed(path)), flush=True)
            if checkpoint_callback:
                checkpoint_callback()
    write_json(out / 'completion.json', {'capped': capped, 'elapsed_seconds': time.monotonic() - start,
                'load_seconds': load_seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})

def metrics(rows, expected):
    valid = [r for r in rows if r['status'] == 'ok']
    if not valid:
        return {'expected': expected, 'attempted': len(rows), 'valid': 0, 'accuracy': None,
                'coverage_adjusted_accuracy': 0.0, 'missed': expected, 'correct': 0,
                'disagreement_rate_valid': None, 'error_rate_missed_wrong': 1.0}
    bins = defaultdict(list)
    for r in valid:
        bins[min(14, int(r['confidence'] * 15))].append(r)
    return {'expected': expected, 'attempted': len(rows), 'valid': len(valid),
            'accuracy': statistics.mean(r['correct'] for r in valid),
            'coverage_adjusted_accuracy': sum(r['correct'] for r in valid) / expected,
            'correct': sum(r['correct'] for r in valid), 'missed': expected - len(valid),
            'disagreement_rate_valid': statistics.mean(not r['correct'] for r in valid),
            'error_rate_missed_wrong': 1 - sum(r['correct'] for r in valid) / expected,
            'nll': statistics.mean(r['nll'] for r in valid), 'brier': statistics.mean(r['brier'] for r in valid),
            'nll_floor_1e_3': statistics.mean(min(r['nll'], -math.log(1e-3)) for r in valid),
            'zero_reference_probability_count': sum(r['probabilities'][r['options'].index(r['label'])] == 0 for r in valid if 'probabilities' in r),
            'ece_15': sum(abs(sum(r['confidence'] - r['correct'] for r in b)) for b in bins.values()) / len(valid),
            'median_latency_ms': statistics.median(r['latency_ms'] for r in valid),
            'errors': len(rows) - len(valid)}

def accuracy_cell(valid_accuracy, adjusted_accuracy, valid, expected):
    if valid == expected:
        return f'{100 * valid_accuracy:.2f}% ({valid}/{expected})'
    first = 'N/A' if valid_accuracy is None else f'{100 * valid_accuracy:.2f}%'
    return f'{first} / {100 * adjusted_accuracy:.2f}% ({valid}/{expected})'

def report(base):
    panel = json.loads((ROOT / 'data/manifests/baseline-panel.json').read_text())
    base = Path(base); result = {}; table = ['| Benchmark | n | PPLX v1-27B | CLEF-flash | CLEF | Jev 1.13.0 |', '|---|---:|---:|---:|---:|---:|']
    for key in MODELS:
        path = base / key / 'predictions.jsonl'
        rows = read_jsonl(path) if path.exists() else []
        if rows:
            cfg = json.loads((base / key / 'config.json').read_text())
            if cfg['panel_sha256'] != sha(ROOT / 'data/manifests/baseline-panel.json'):
                raise ValueError('Results belong to another panel: ' + key)
            expected_rows = {r['id']: r for r in load_panel()}
            for row in rows:
                original = expected_rows[row['id']]
                if row['suite'] != original['suite'] or row['label'] != original['label']:
                    raise ValueError('Prediction provenance mismatch: ' + row['id'])
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate prediction IDs')
        result[key] = {}
        for suite, spec in panel['suites'].items():
            selected = [r for r in rows if r['suite'] == suite]
            m = metrics(selected, spec['count'])
            if suite == 'BBH' and selected:
                m['subtasks'] = {s: metrics([r for r in selected if r['subtask'] == s], n) for s, n in spec['subtasks'].items()}
                m['task_macro_coverage_adjusted_accuracy'] = statistics.mean(x['coverage_adjusted_accuracy'] for x in m['subtasks'].values())
                if all(x['accuracy'] is not None for x in m['subtasks'].values()):
                    m['task_macro_accuracy'] = statistics.mean(x['accuracy'] for x in m['subtasks'].values())
            result[key][suite] = m
    for suite, spec in panel['suites'].items():
        cells = []
        for key in ['pplx', 'clef-flash', 'clef', 'jev']:
            m = result[key][suite]; value = m.get('task_macro_accuracy', m['accuracy'])
            adjusted = m.get('task_macro_coverage_adjusted_accuracy', m['coverage_adjusted_accuracy'])
            cells.append('Pending' if m['attempted'] == 0 else accuracy_cell(value, adjusted, m['valid'], m['expected']))
        table.append(f'| {suite} | {spec["count"]} | ' + ' | '.join(cells) + ' |')
    cells = []
    for key in ['pplx', 'clef-flash', 'clef', 'jev']:
        suites = result[key]
        if all(m['attempted'] == m['expected'] and m['accuracy'] is not None for m in suites.values()):
            value = statistics.mean(m.get('task_macro_accuracy', m['accuracy']) for m in suites.values())
            cells.append(f'{100*value:.2f}%')
            result[key]['overall'] = {'suite_macro_accuracy': value}
        else:
            cells.append('Pending')
    cells = []
    for key in ['pplx', 'clef-flash', 'clef', 'jev']:
        suites = {s: result[key][s] for s in panel['suites']}
        if all(m['attempted'] == m['expected'] for m in suites.values()):
            values = []
            for suite, m in suites.items():
                if suite == 'BBH':
                    values.append(statistics.mean(x['coverage_adjusted_accuracy'] if x['valid'] else 0 for x in m['subtasks'].values()))
                else:
                    values.append(m['coverage_adjusted_accuracy'] if m['valid'] else 0)
            value = statistics.mean(values)
            cells.append(f'{100*value:.2f}%')
            result[key].setdefault('overall', {})['coverage_adjusted_suite_macro_accuracy'] = value
        else:
            cells.append('Pending')
    cells = []
    for key in ['pplx', 'clef-flash', 'clef', 'jev']:
        suites = {s: result[key][s] for s in panel['suites']}
        overall = result[key].get('overall', {})
        valid = overall.get('suite_macro_accuracy')
        adjusted = overall.get('coverage_adjusted_suite_macro_accuracy')
        if adjusted is None:
            cells.append('Pending')
        elif all(m['valid'] == m['expected'] for m in suites.values()):
            cells.append(f'{100 * adjusted:.2f}%')
        else:
            first = 'N/A' if valid is None else f'{100 * valid:.2f}%'
            cells.append(f'{first} / {100 * adjusted:.2f}%')
    table.append('| **11-suite macro** | **2111** | ' + ' | '.join(cells) + ' |')
    write_json(base / 'metrics.json', result)
    calibration = ['| Model | Benchmark | NLL (1e-12) | NLL (1e-3) | Zero gold p | Brier | ECE (15 bins) | Median ms |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for key in MODELS:
        for suite in panel['suites']:
            m = result[key][suite]
            if m['valid']:
                calibration.append(f'| {key} | {suite} | {m["nll"]:.4f} | {m["nll_floor_1e_3"]:.4f} | {m["zero_reference_probability_count"]} | {m["brier"]:.4f} | {m["ece_15"]:.4f} | {m["median_latency_ms"]:.1f} |')
    (base / 'results.md').write_text('# Sampled baselines — measured results\n\n200 label-blind examples per benchmark; all 111 JevBench hard examples. Total: 2,111 shared decisions. These are not full-benchmark scores.\n\n'+ '\n'.join(table) + '\n\nPPLX is perplexity-ai/pplx-decider-v1-27b (v1), pinned to d8c3bd18e31dc1804433d38528b3e770019bf00c. No v1.1 results are included.\n\nCells show one accuracy when all predictions are valid. Only cases with misses show **accuracy excluding misses / accuracy counting misses as wrong**. Parentheses give valid/expected coverage. A miss is any rejected, failed, refused, or unattempted panel ID. A valid prediction disagreeing with gold counts as wrong in both versions. BBH uses its subtask macro-average in both versions. Macro scores are equally weighted across 11 benchmarks. Pending is not zero. Calibration is measured on valid predictions only. NLL is shown at two numerical floors because reported zero probabilities and API rounding affect confident-error penalties. Local and API latencies are different measurement conditions.\n\n' + '\n'.join(calibration) + '\n')
    if (base / 'random-joint-qwen38-27b/config.json').exists():
        from random_joint_baseline import merge_table
        merge_table(base)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze','jev','report','local'])
    p.add_argument('--out', default=str(ROOT / 'outputs/baselines-20261014'))
    p.add_argument('--model', choices=list(MODELS)); p.add_argument('--limit', type=int, default=200)
    p.add_argument('--cap-seconds', type=int, default=3600)
    args = p.parse_args()
    if args.action == 'freeze': freeze(args.limit)
    elif args.action == 'jev': asyncio.run(run_jev(Path(args.out) / 'jev', args.cap_seconds))
    elif args.action == 'local': run_local(args.model, Path(args.out) / args.model, args.cap_seconds)
    else: report(args.out)
