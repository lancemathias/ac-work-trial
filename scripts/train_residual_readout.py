"""Paired residual-readout order ablation; resumable frozen final-token caches."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from residual_code_readout import ResidualReadout, encode, semantic_logits, verify_codes
from head_probe_io import prepare_run, seal, write_json
from training_data import canonical, digest, load_splits

ROOT = Path(__file__).resolve().parents[1]


def atomic_save(value, path):
    path = Path(path)
    torch.save(value, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)


def extract(root, prepared, backbone, tokenizer, checkpoint, deadline):
    """Each committed shard binds exact inputs, order, targets, and tensor bytes."""
    root = Path(root)
    identity = hashlib.sha256(canonical(prepared).encode()).hexdigest()
    old = prepare_run(root, {'encodings_sha256': identity, 'count': len(prepared), 'batch_size': 4})
    if old:
        return read_features(root)
    ordered = sorted(prepared, key=lambda x: (len(x['input_ids']), x['id']))
    text = backbone.model
    if hasattr(text, 'language_model'):
        text = text.language_model
    device = next(backbone.parameters()).device
    began = time.monotonic()
    for start in range(0, len(ordered), 128):
        target = root / f'shard-{start:06d}.pt'
        receipt = target.with_suffix('.json')
        group = ordered[start:start + 128]
        expected = hashlib.sha256(canonical(group).encode()).hexdigest()
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value != {'inputs_sha256': expected, 'tensor_sha256': digest(target)}:
                raise ValueError('Feature shard identity or bytes changed')
            continue
        if time.time() + 180 > deadline:
            raise TimeoutError('Reservation nearly over; cached progress preserved')
        vectors = []
        for offset in range(0, len(group), 4):
            batch = group[offset:offset + 4]
            lengths = [len(x['input_ids']) for x in batch]
            ids = torch.full((len(batch), max(lengths)), tokenizer.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                ids[i, :lengths[i]] = torch.tensor(row['input_ids'], device=device)
                mask[i, :lengths[i]] = 1
            with torch.no_grad():
                hidden = text(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
                last = hidden[torch.arange(len(batch), device=device), torch.tensor(lengths, device=device) - 1]
                if not torch.isfinite(last).all():
                    raise ValueError('Nonfinite frozen features')
                vectors.append(last.cpu())
            del hidden, last, ids, mask
        atomic_save({'features': torch.cat(vectors),
                     'items': [{k: v for k, v in x.items() if k != 'input_ids'} for x in group]}, target)
        write_json(receipt, {'inputs_sha256': expected, 'tensor_sha256': digest(target)})
        checkpoint()
        print(canonical({'event': 'residual_feature_progress', 'role': root.name,
                         'records': min(start + 128, len(ordered)), 'total': len(ordered),
                         'seconds': time.monotonic() - began}), flush=True)
    write_json(root / 'result.json', {'status': 'complete', 'request': json.loads((root / 'request.json').read_text())})
    seal(root)
    checkpoint()
    return read_features(root)


def read_features(root):
    chunks = [torch.load(p, map_location='cpu', weights_only=True) for p in sorted(Path(root).glob('shard-*.pt'))]
    items = [item for chunk in chunks for item in chunk['items']]
    if len(items) != json.loads((Path(root) / 'request.json').read_text())['count'] or len({x['id'] for x in items}) != len(items):
        raise ValueError('Feature cache missing or duplicate records')
    # Pair minibatches by record identity, independent of permutation-dependent prompt lengths.
    order = sorted(range(len(items)), key=lambda i: items[i]['id'])
    return torch.cat([c['features'] for c in chunks])[order], [items[i] for i in order]


def predict(head, cached, originals):
    features, items = cached
    results = []
    device = next(head.parameters()).device
    head.eval()
    with torch.no_grad():
        for start in range(0, len(items), 64):
            group = items[start:start + 64]
            counts = torch.tensor([len(x['option_ids']) for x in group], device=device)
            values = head(features[start:start + len(group)].to(device), counts).cpu()
            for item, logits in zip(group, values, strict=True):
                ids = originals[item['id']]
                results.append(semantic_logits(item, logits[:len(ids)], ids))
    return results


def train_epoch(head, optimizer, cached, cfg, epoch, total_steps):
    features, items = cached
    device = next(head.parameters()).device
    generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
    indices = torch.randperm(len(items), generator=generator)
    head.train()
    loss_sum = 0.0
    steps = math.ceil(len(items) / cfg['effective_batch_decisions'])
    warmup = max(1, int(total_steps * cfg['warmup_fraction']))
    for batch, start in enumerate(range(0, len(items), cfg['effective_batch_decisions'])):
        take = indices[start:start + cfg['effective_batch_decisions']]
        group = [items[i] for i in take.tolist()]
        step = epoch * steps + batch
        scale = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for params in optimizer.param_groups:
            params['lr'] = cfg['learning_rate'] * scale
        counts = torch.tensor([len(x['option_ids']) for x in group], device=device)
        targets = torch.tensor([x['target_index'] for x in group], device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = head(features[take].to(device), counts)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(group)
    return loss_sum / len(items)


def order_metrics(native, changed):
    by_id = {x['id']: x for x in changed}
    flips, distances = [], []
    for base in native:
        other = by_id[base['id']]
        if base['option_ids'] != other['option_ids']:
            raise ValueError('Order comparison requires semantic alignment')
        a, b = [torch.tensor(x['logits'], dtype=torch.float64).softmax(-1) for x in [base, other]]
        flips.append(int(a.argmax() != b.argmax()))
        distances.append(float((a - b).abs().sum() / 2))
    return {'decisions': len(flips), 'argmax_flip_rate': sum(flips) / len(flips),
            'mean_total_variation': sum(distances) / len(distances)}


def run(out, config_path, source_identity, checkpoint=lambda: None, deadline=float('inf')):
    from train_joint import metrics, fit_temperature, slice_metrics
    from run_baselines import load_panel, record, score
    from evaluate_trained_joint import answer_body
    from vendor.clef_joint_schema import question_options
    from resident_backbone import get_model, get_processor
    from kernel_runtime import configure_optimized_runtime
    cfg = json.loads(Path(config_path).read_text())
    out = Path(out)
    mp = ROOT / cfg['split_manifest']
    if digest(mp) != cfg['split_manifest_sha256']:
        raise ValueError('Original v3 split manifest changed')
    splits, manifest = load_splits(mp)
    if manifest['source_dataset_sha256'] != cfg['dataset']['sha256']:
        raise ValueError('Require exact v3 dataset')
    request = dict(config=cfg, source=source_identity, codes_sha256=digest(ROOT / 'configs/readout-mlp/codes-255.json'))
    prior = prepare_run(out, request)
    if prior:
        return prior
    write_json(out / 'config.json', request)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    runtime = configure_optimized_runtime('both')
    write_json(out / 'kernel-runtime.json', runtime)
    import wandb
    tracker = wandb.init(project=cfg['wandb_project'], name=out.name, config=request, job_type='residual-readout-order-ablation')
    write_json(out / 'wandb-run.json', {'id': tracker.id, 'url': tracker.url})
    processor = get_processor(cfg)
    code_ids = verify_codes(processor.tokenizer, cfg['backbone_id'])
    backbone = get_model(cfg)
    frozen = [(p, p._version) for p in backbone.parameters()]
    output = backbone.get_output_embeddings()
    indices = torch.tensor(code_ids, device=output.weight.device)
    weight = output.weight[indices].detach().float()
    bias = getattr(output, 'bias', None)
    bias = None if bias is None else bias[indices].detach().float()
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    initial = ResidualReadout(weight, bias, cfg['head']['width']).cuda()
    initial_state = copy.deepcopy(initial.state_dict())
    heads, optimizers = {}, {}
    for variant in cfg['variants']:
        head = copy.deepcopy(initial)
        groups = [{'params': [p for p in head.parameters() if p.ndim >= 2], 'weight_decay': cfg['weight_decay']},
                  {'params': [p for p in head.parameters() if p.ndim < 2], 'weight_decay': 0.0}]
        heads[variant] = head
        optimizers[variant] = torch.optim.AdamW(groups, lr=cfg['learning_rate'], betas=tuple(cfg['adamw_betas']), eps=cfg['adamw_eps'])
    del initial
    atomic_save(initial_state, out / 'initial-head.pt')
    write_json(out / 'architecture.json', {'trainable_parameters': sum(p.numel() for p in heads['fixed'].parameters()),
                                         'frozen_code_rows': 255, 'hidden_width': 1024, 'same_initialization': True})
    checkpoint()
    cache = out / 'features'
    def prepared(rows, training=True, order='native', epoch=0):
        return [encode(processor.tokenizer, row, training, order, cfg['seed'], epoch, cfg['max_input_tokens']) for row in rows]
    def features(name, rows, training=True, order='native', epoch=0):
        return extract(cache / name, prepared(rows, training, order, epoch), backbone, processor.tokenizer, checkpoint, deadline)
    def original_ids(rows, training=True):
        return {r['id']: [k for k, _ in question_options(r['request']['questions']['q'] if training else r['question'])] for r in rows}
    dev = features('dev-native', splits['dev'])
    dev_ids = original_ids(splits['dev'])
    fixed = features('train-fixed', splits['train'])
    total_steps = math.ceil(len(splits['train']) / cfg['effective_batch_decisions']) * cfg['epochs']
    # Record the untrained randomly initialized residual head before fitting.
    if not (out / 'initial-random-head-dev.json').exists():
        write_json(out / 'initial-random-head-dev.json', metrics(predict(heads['fixed'], dev, dev_ids)))
    for epoch in range(cfg['epochs']):
        shuffled = None
        for variant in cfg['variants']:
            folder = out / variant
            folder.mkdir(exist_ok=True)
            complete = folder / f'epoch-{epoch + 1}.pt'
            if complete.exists():
                saved = torch.load(complete, weights_only=True, map_location='cuda')
                heads[variant].load_state_dict(saved['head'])
                optimizers[variant].load_state_dict(saved['optimizer'])
                continue
            if variant == 'shuffled' and shuffled is None:
                shuffled = features(f'train-shuffled-epoch-{epoch + 1}', splits['train'], order='shuffle', epoch=epoch + 1)
            loss = train_epoch(heads[variant], optimizers[variant], fixed if variant == 'fixed' else shuffled, cfg, epoch, total_steps)
            dev_predictions = predict(heads[variant], dev, dev_ids)
            values = metrics(dev_predictions)
            write_json(folder / f'epoch-{epoch + 1}.json', {'epoch': epoch + 1, 'train_nll': loss, 'dev': values})
            best_path = folder / 'best.json'
            best = json.loads(best_path.read_text()) if best_path.exists() else None
            if best is None or values['nll'] < best['dev']['nll']:
                atomic_save(heads[variant].state_dict(), folder / 'best-head.pt')
                write_json(best_path, {'epoch': epoch + 1, 'dev': values})
            atomic_save({'head': heads[variant].state_dict(), 'optimizer': optimizers[variant].state_dict()}, complete)
            checkpoint()
            tracker.log({variant + '/epoch': epoch + 1, variant + '/train_nll': loss,
                         **{variant + '/dev_' + k: v for k, v in values.items()}})
            print(canonical({'event': 'residual_epoch', 'variant': variant, 'epoch': epoch + 1, 'train_nll': loss, 'dev': values}), flush=True)
        del shuffled
    calibration = features('calibration-native', splits['calibration'])
    calibration_ids = original_ids(splits['calibration'])
    temperatures = {}
    for variant in cfg['variants']:
        head = heads[variant]
        head.load_state_dict(torch.load(out / variant / 'best-head.pt', weights_only=True, map_location='cuda'))
        fitted = fit_temperature(predict(head, calibration, calibration_ids))
        temperatures[variant] = fitted['temperature']
        write_json(out / variant / 'temperature.json', fitted)
        write_json(out / variant / 'dev.json', {'raw': metrics(predict(head, dev, dev_ids)),
                                              'calibrated': metrics(predict(head, dev, dev_ids), fitted['temperature'])})
    checkpoint()
    # All head selection and temperature fitting finish before benchmark labels are used.
    rows = load_panel()
    originals = {r['id']: r for r in rows}
    test_ids = original_ids(rows, False)
    summaries = {v: {} for v in cfg['variants']}
    test_predictions = {}
    for order in cfg['evaluation_orders']:
        cached = features('test-' + order, rows, False, order, 1000000)
        for variant in cfg['variants']:
            predictions = predict(heads[variant], cached, test_ids)
            test_predictions[variant, order] = predictions
            folder = out / variant / ('test-' + order)
            folder.mkdir(exist_ok=True)
            (folder / 'logits.jsonl').write_text(''.join(canonical(x) + '\n' for x in predictions))
            for name, temp in [('raw', 1.0), ('calibrated', temperatures[variant])]:
                values = []
                for item in predictions:
                    row = originals[item['id']]
                    body = answer_body(row['question'], item['option_ids'], (torch.tensor(item['logits'], dtype=torch.float64) / temp).softmax(-1).tolist())
                    values.append(record(row, dict(score(row, body), response=body)))
                (folder / (name + '-predictions.jsonl')).write_text(''.join(canonical(x) + '\n' for x in values))
            from report_trained_joint import report
            unchanged = all(not p.requires_grad and p.grad is None and p._version == version for p, version in frozen)
            if not unchanged:
                raise RuntimeError('Backbone was modified')
            write_json(folder / 'config.json', {'panel_sha256': digest(ROOT / 'data/manifests/baseline-panel.json'),
                                               'temperature': temperatures[variant],
                                               'checkpoint_sha256': digest(out / variant / 'best-head.pt')})
            write_json(folder / 'completion.json', {'status': 'complete', 'parameters_unchanged': True})
            summaries[variant][order] = report(folder, label=f'{variant} residual MLP; {order} options')
            checkpoint()
    for variant in cfg['variants']:
        native = test_predictions[variant, 'native']
        summaries[variant]['order_sensitivity'] = {order: order_metrics(native, test_predictions[variant, order]) for order in ['reverse', 'shuffle']}
    unchanged = all(not p.requires_grad and p.grad is None and p._version == version for p, version in frozen)
    if not unchanged:
        raise RuntimeError('Backbone was modified')
    write_json(out / 'comparison.json', summaries)
    tracker.summary['comparison'] = summaries
    tracker.finish()
    result = {'status': 'complete', 'request': request, 'comparison': summaries, 'backbone_parameters_unchanged': True}
    write_json(out / 'result.json', result)
    seal(out)
    checkpoint()
    return result
