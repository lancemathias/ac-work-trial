"""Frozen original Qwen + three untrained Clef JointSchemaHead initializations."""
import argparse
import json
import math
import statistics
from pathlib import Path

from run_baselines import ROOT, canonical, load_panel, metrics, payload, read_jsonl, record, score, sha, write_json

SEEDS = (20261014, 20261015, 20261016)
BACKBONE = ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
CLEF = ('Cloudflare/clef', '0b331204bb13fbd2ca93a64df1956f5b55478ce5')
SOURCE_HASH = '0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'
HEAD_CONFIG = dict(hidden_size=5120, width=1024, routing_layers=2, layers=4, heads=16, feedforward=4096)
RUN = 'random-joint-qwen38-27b'


def evaluate(out, cap_seconds=3500, checkpoint_callback=None):
    import hashlib
    import random
    import sys
    import time
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    started = time.monotonic()
    out = Path(out)
    if (out / 'config.json').exists():
        raise FileExistsError('Use a fresh output directory; do not mix head initializations/runs')
    out.mkdir(parents=True, exist_ok=True)
    rows = load_panel()
    cfg = dict(kind='untrained_random_joint_head', seeds=list(SEEDS), backbone_id=BACKBONE[0],
               backbone_revision=BACKBONE[1], head_source=CLEF[0], head_source_revision=CLEF[1],
               head_source_sha256=SOURCE_HASH, head_config=HEAD_CONFIG,
               panel_sha256=sha(ROOT / 'data/manifests/baseline-panel.json'),
               runner_sha256=sha(__file__), scoring_runner_sha256=sha(ROOT / 'scripts/run_baselines.py'),
               torch=torch.__version__, transformers=__import__('transformers').__version__,
               gpu=torch.cuda.get_device_name(), precision='bfloat16', max_input_tokens=8192,
               batch_size=1, trained=False, backbone_frozen=True, calibration='T=1; no fitting',
               initialization='Native Clef constructor in CPU FP32, seeded before each head; cast to BF16',
               lexical_prior='Original frozen Qwen output embeddings and native Clef prior retained',
               cap_seconds=cap_seconds, expected_per_seed=len(rows))
    write_json(out / 'config.json', cfg)
    if checkpoint_callback:
        checkpoint_callback()
    try:
        source = hf_hub_download(*CLEF[:1], filename='joint_schema_model.py', revision=CLEF[1], cache_dir='/models/hub')
        if sha(source) != SOURCE_HASH:
            raise ValueError('Pinned upstream implementation changed')
        sys.path.insert(0, str(Path(source).parent))
        from joint_schema_model import JointSchemaHead, collate_records, encode_record

        torch.set_num_threads(8)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            BACKBONE[0], revision=BACKBONE[1], cache_dir='/models/hub',
            dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda'})
        backbone.requires_grad_(False).eval()
        processor = AutoProcessor.from_pretrained(BACKBONE[0], revision=BACKBONE[1], cache_dir='/models/hub')
        params = list(backbone.parameters())
        versions = [p._version for p in params]
        if any(p.requires_grad for p in params):
            raise ValueError('Backbone not frozen')
        heads = []
        for seed in SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            head = JointSchemaHead(**HEAD_CONFIG).float().cpu().eval()
            seed_out = out / f'seed-{seed}'
            seed_out.mkdir()
            save_file({k: v.detach().contiguous() for k,v in head.state_dict().items()}, str(seed_out / 'initial-head.safetensors'))
            write_json(seed_out / 'config.json', dict(cfg, seed=seed,
                initial_head_sha256=sha(seed_out / 'initial-head.safetensors'),
                head_parameters=sum(p.numel() for p in head.parameters())))
            heads.append(head.to(device='cuda', dtype=torch.bfloat16))
        cfg['head_parameters'] = sum(p.numel() for p in heads[0].parameters())
        cfg['load_seconds'] = time.monotonic() - started
        write_json(out / 'config.json', cfg)
        if checkpoint_callback:
            checkpoint_callback()
        text_model = backbone.model
        if hasattr(text_model, 'language_model'):
            text_model = text_model.language_model
        output_embeddings = backbone.get_output_embeddings().weight
        processed = 0
        for row in rows:
            if time.monotonic() - started >= cap_seconds:
                break
            results = []
            began = time.monotonic()
            try:
                request = payload(row, BACKBONE[0])
                encoded = encode_record(processor.tokenizer, request, max_length=10_000_000, processor=processor)
                if len(encoded.input_ids) > 8192:
                    raise ValueError('Input exceeds common 8192-token limit; no truncation')
                batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device('cuda'))
                with torch.inference_mode():
                    hidden = text_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False, return_dict=True).last_hidden_state
                    torch.cuda.synchronize()
                    backbone_ms = 1000 * (time.monotonic() - began)
                    for head in heads:
                        head_started = time.monotonic()
                        logits = head(hidden, batch['input_ids'], batch['attention_mask'], batch['records'], output_embeddings)[0][0]
                        probabilities = logits.float().softmax(-1).cpu().tolist()
                        torch.cuda.synchronize()
                        head_ms = 1000 * (time.monotonic() - head_started)
                        question = row['question']
                        distribution = dict(zip(encoded.questions[0].option_ids, probabilities, strict=True))
                        # Preserve full precision; the released serializer rounds to four decimals.
                        answer = {'type': question['type']}
                        if question['type'] == 'noul':
                            answer['noul'] = distribution['true']
                        else:
                            answer['probabilities'] = distribution
                        body = {'answers': {'q': answer}}
                        results.append(dict(score(row, body), response=body,
                            input_tokens=len(encoded.input_ids), backbone_ms=backbone_ms,
                            head_ms=head_ms, latency_ms=backbone_ms + head_ms))
                del hidden, batch
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results = [dict(status='error', error='OutOfMemoryError') for _ in SEEDS]
            except Exception as exc:
                results = [dict(status='error', error=type(exc).__name__, detail=str(exc)[:250]) for _ in SEEDS]
            for seed, result in zip(SEEDS, results, strict=True):
                with (out / f'seed-{seed}' / 'predictions.jsonl').open('a') as stream:
                    stream.write(canonical(record(row, dict(result, seed=seed))) + '\n')
            processed += 1
            if processed % 50 == 0:
                print(f'Frozen Qwen + 3 random heads: {processed}/{len(rows)}; {time.monotonic()-started:.1f}s', flush=True)
                if checkpoint_callback:
                    checkpoint_callback()
        unchanged = all(p._version == version and not p.requires_grad for p, version in zip(params, versions, strict=True))
        if not unchanged:
            raise ValueError('Backbone parameters mutated')
        completion = dict(status='complete' if processed == len(rows) else 'capped', processed_per_seed=processed,
            expected_per_seed=len(rows), backbone_parameters_unchanged=unchanged,
            elapsed_seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(out / 'completion.json', completion)
        print(completion, flush=True)
        return completion
    except Exception as exc:
        write_json(out / 'completion.json', dict(status='failed', error=type(exc).__name__, detail=str(exc)[:500], elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if checkpoint_callback:
            checkpoint_callback()


def summarize(base):
    base = Path(base)
    folder = base / RUN
    if not (folder / 'config.json').exists():
        return None
    cfg = json.loads((folder / 'config.json').read_text())
    if cfg['seeds'] != list(SEEDS) or cfg['panel_sha256'] != sha(ROOT / 'data/manifests/baseline-panel.json'):
        raise ValueError('Run config does not match seeds/panel')
    panel = json.loads((ROOT / 'data/manifests/baseline-panel.json').read_text())
    expected = {r['id']: r for r in load_panel()}
    per_seed = {}
    for seed in SEEDS:
        path = folder / f'seed-{seed}/predictions.jsonl'
        rows = read_jsonl(path) if path.exists() else []
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate predictions')
        for row in rows:
            source = expected[row['id']]
            if row['seed'] != seed or row['suite'] != source['suite'] or row['label'] != source['label']:
                raise ValueError('Prediction provenance mismatch')
            if row['status'] == 'ok':
                rebuilt = score(source, row['response'])
                if (row['correct'] != rebuilt['correct']
                    or len(row['probabilities']) != len(rebuilt['probabilities'])
                    or any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
                           for a,b in zip(row['probabilities'],rebuilt['probabilities'],strict=True))
                    or any(not math.isclose(row[k],rebuilt[k],rel_tol=1e-12,abs_tol=1e-12)
                           for k in ['nll','brier'])):
                    raise ValueError('Stored scores do not match probability trace')
        suites = {}
        for suite, spec in panel['suites'].items():
            selected = [r for r in rows if r['suite'] == suite]
            m = metrics(selected, spec['count'])
            m.setdefault('coverage_adjusted_accuracy', 0.0)
            if suite == 'BBH':
                subtasks = [metrics([r for r in selected if r['subtask']==sub],n) for sub,n in spec['subtasks'].items()]
                if all(x['valid'] for x in subtasks):
                    m['accuracy'] = statistics.mean(x['accuracy'] for x in subtasks)
                else:
                    m['accuracy'] = None
                m['coverage_adjusted_accuracy'] = statistics.mean(x.get('coverage_adjusted_accuracy',0) for x in subtasks)
            suites[suite] = m
        per_seed[str(seed)] = suites
    aggregate = {}
    for suite in panel['suites']:
        ms = [per_seed[str(seed)][suite] for seed in SEEDS]
        row = dict(complete=all(m['valid']==m['expected'] for m in ms),
                   valid_per_seed=[m['valid'] for m in ms], attempted_per_seed=[m['attempted'] for m in ms], expected=ms[0]['expected'])
        for metric in ['accuracy','nll','brier','ece_15','coverage_adjusted_accuracy']:
            values = [m.get(metric) for m in ms]
            if all(v is not None for v in values):
                row[metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), per_seed=values)
        aggregate[suite] = row
    macros=[]
    if all(r['complete'] for r in aggregate.values()):
        macros=[statistics.mean(per_seed[str(seed)][s]['accuracy'] for s in panel['suites']) for seed in SEEDS]
    all_attempted = all(all(n==r['expected'] for n in r['attempted_per_seed']) for r in aggregate.values())
    adjusted = [statistics.mean(per_seed[str(seed)][s]['coverage_adjusted_accuracy'] for s in panel['suites']) for seed in SEEDS] if all_attempted else []
    valid_only = [statistics.mean(per_seed[str(seed)][s]['accuracy'] for s in panel['suites']) for seed in SEEDS] if all_attempted and all(r.get('accuracy') is not None for r in aggregate.values()) else []
    result=dict(seeds=list(SEEDS), per_seed=per_seed, suites=aggregate,
        overall=None if not macros else dict(mean=statistics.mean(macros),sample_sd=statistics.stdev(macros),per_seed=macros),
        valid_only_overall=None if not valid_only else dict(mean=statistics.mean(valid_only),sample_sd=statistics.stdev(valid_only),per_seed=valid_only),
        coverage_adjusted_overall=None if not adjusted else dict(mean=statistics.mean(adjusted),sample_sd=statistics.stdev(adjusted),per_seed=adjusted))
    write_json(folder / 'metrics.json', result)
    return result


def merge_table(base):
    base = Path(base)
    result = summarize(base)
    if result is None:
        return
    path=base / 'results.md'
    lines=path.read_text().splitlines()
    heading='Frozen Qwen3.8-27B + random joint head (3 seeds)'
    panel=json.loads((ROOT / 'data/manifests/baseline-panel.json').read_text())
    in_table=False
    for i,line in enumerate(lines):
        if line.startswith('| Benchmark | n |'):
            in_table=True
            cells=line.split('|')
            if heading in line:
                continue
            lines[i]=line+f' {heading} |'
        elif in_table and line.startswith('|'):
            cells=[c.strip() for c in line.split('|')[1:-1]]
            if len(cells)>6:
                cells=cells[:6]
                line='| '+' | '.join(cells)+' |'
            name=cells[0]
            if name.startswith('---'):
                value='---:'
            elif name in panel['suites']:
                r=result['suites'][name]
                m=r.get('accuracy')
                value='Pending' if m is None else f'{100*m["mean"]:.2f}% ± {100*m["sample_sd"]:.2f} ({"/".join(map(str,r["valid_per_seed"]))}; n={r["expected"]})'
                if m is not None and not r['complete'] and 'coverage_adjusted_accuracy' in r:
                    a=r['coverage_adjusted_accuracy']
                    value=f'{100*m["mean"]:.2f}% ± {100*m["sample_sd"]:.2f} / {100*a["mean"]:.2f}% ± {100*a["sample_sd"]:.2f} ({"/".join(map(str,r["valid_per_seed"]))}; n={r["expected"]})'
            elif name=='**11-suite macro**':
                m=result['overall']
                value='Incomplete' if m is None else f'{100*m["mean"]:.2f}% ± {100*m["sample_sd"]:.2f}'
                if m is None and result['valid_only_overall'] is not None and result['coverage_adjusted_overall'] is not None:
                    v=result['valid_only_overall'];a=result['coverage_adjusted_overall']
                    value=f'{100*v["mean"]:.2f}% ± {100*v["sample_sd"]:.2f} / {100*a["mean"]:.2f}% ± {100*a["sample_sd"]:.2f}'
            elif name=='**Coverage-adjusted macro**':
                complete=all(all(n==r['expected'] for n in r['attempted_per_seed']) for r in result['suites'].values())
                values=[statistics.mean(result['per_seed'][str(seed)][s]['coverage_adjusted_accuracy'] for s in panel['suites']) for seed in SEEDS] if complete else []
                value='Pending' if not values else f'{100*statistics.mean(values):.2f}% ± {100*statistics.stdev(values):.2f}'
            else:
                continue
            lines[i]='| '+' | '.join(cells[:6]+[value])+' |'
        elif in_table:
            break
    path.write_text('\n'.join(lines)+'\n')


def export_artifacts(base, profile='ac-worktrial', environment='worktrial-lance'):
    import subprocess
    base = Path(base)
    target = base / RUN
    scratch = ROOT / 'work/random-joint-export'
    scratch.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'completion.json']
    files += [f'seed-{seed}/{name}' for seed in SEEDS for name in ['config.json', 'predictions.jsonl']]
    for name in files:
        temporary = scratch / name.replace('/', '__')
        subprocess.run([str(ROOT/'.venv/bin/modal'), 'volume', 'get', '--profile', profile,
            '--env', environment, 'ac-worktrial-baselines', f'baselines-20261014/{RUN}/{name}',
            str(temporary), '--force'], check=True)
        destination = target / name
        data = temporary.read_bytes()
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different existing evidence: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default=str(ROOT/'outputs/baselines-20261014'))
    p.add_argument('--export', action='store_true')
    args=p.parse_args()
    if args.export:
        export_artifacts(args.base)
    merge_table(args.base)
