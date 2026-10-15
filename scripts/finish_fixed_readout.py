"""Finish only the fixed condition from committed features and optimizer state."""
import json
import math
import time
from pathlib import Path
import torch
from head_probe_io import prepare_run, seal, write_json
from training_data import digest, load_splits, canonical
from train_residual_readout import atomic_save, read_features, predict, train_epoch, extract
from residual_code_readout import ResidualReadout, encode, verify_codes

ROOT = Path(__file__).resolve().parents[1]


def verified_cache(root):
    root = Path(root)
    if json.loads((root / 'result.json').read_text())['status'] != 'complete':
        raise ValueError('Cache incomplete')
    hashes = json.loads((root / 'checksums.json').read_text())
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('Cache checksum mismatch')
    return read_features(root)


def best_epoch(records):
    # Preserve the project's user-directed accuracy selection, earliest on ties.
    return max(records, key=lambda r: (r['dev']['accuracy'], -r['epoch']))['epoch']


def run(parent, out, source, checkpoint, deadline):
    from train_joint import metrics, fit_temperature
    from run_baselines import load_panel, record, score
    from evaluate_trained_joint import answer_body
    from vendor.clef_joint_schema import question_options
    import wandb
    parent, out = Path(parent), Path(out)
    provenance = json.loads((parent / 'config.json').read_text())
    cfg = provenance['config']
    if cfg['head']['width'] != 1024 or cfg['backbone_id'] != 'Qwen/Qwen3.8-27B':
        raise ValueError('Unexpected parent experiment')
    if provenance['codes_sha256'] != digest(ROOT / 'configs/readout-mlp/codes-255.json'):
        raise ValueError('Answer codes changed')
    mp = ROOT / cfg['split_manifest']
    if digest(mp) != cfg['split_manifest_sha256']:
        raise ValueError('v3 split changed')
    splits, manifest = load_splits(mp)
    if manifest['source_dataset_sha256'] != cfg['dataset']['sha256']:
        raise ValueError('Dataset changed')
    request = {'parent': parent.name, 'parent_request_sha256': digest(parent / 'request.json'),
               'source': source, 'variant': 'fixed', 'epochs': cfg['epochs'],
               'selection': 'maximum native dev accuracy, earliest epoch on ties', 'evaluation_orders': ['native']}
    prior = prepare_run(out, request)
    if prior:
        return prior
    write_json(out / 'config.json', {'config': cfg, 'continuation': request})
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    train = verified_cache(parent / 'features/train-fixed')
    dev = verified_cache(parent / 'features/dev-native')
    for role, cached in [('train', train), ('dev', dev)]:
        if {x['id'] for x in cached[1]} != {x['id'] for x in splits[role]}:
            raise ValueError('Cache partition differs')
    references = {role: {'path': str(parent / 'features' / name), 'checksums_sha256': digest(parent / 'features' / name / 'checksums.json')}
                  for role, name in [('train', 'train-fixed'), ('dev', 'dev-native')]}
    write_json(out / 'cache-references.json', references)
    epochs = [json.loads(p.read_text()) for p in sorted((parent / 'fixed').glob('epoch-*.json'))
              if p.with_suffix('.pt').exists()]
    for epoch in epochs:
        write_json(out / f"epoch-{epoch['epoch']}.json", epoch)
    latest = max((x['epoch'] for x in epochs), default=0)
    state = torch.load(parent / 'fixed' / f'epoch-{latest}.pt', weights_only=True, map_location='cuda') if latest else None
    initial = torch.load(parent / 'initial-head.pt', weights_only=True, map_location='cuda')
    head = ResidualReadout(initial['code_weight'], initial.get('code_bias'), 1024).cuda()
    head.load_state_dict(initial if state is None else state['head'])
    groups = [{'params': [p for p in head.parameters() if p.ndim >= 2], 'weight_decay': cfg['weight_decay']},
              {'params': [p for p in head.parameters() if p.ndim < 2], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], betas=tuple(cfg['adamw_betas']), eps=cfg['adamw_eps'])
    if state:
        optimizer.load_state_dict(state['optimizer'])
    tracker = wandb.init(project=cfg['wandb_project'], name=out.name, config=request, job_type='fixed-readout-cache-continuation')
    write_json(out / 'wandb-run.json', {'id': tracker.id, 'url': tracker.url})
    write_json(out / 'resume.json', {'completed_parent_epochs': latest, 'remaining_epochs': cfg['epochs'] - latest,
                                    'parent_checkpoint_sha256': digest(parent / 'fixed' / f'epoch-{latest}.pt') if latest else None,
                                    'training_backbone_forwards': 0})
    checkpoint()
    ids = {x['id']: x['option_ids'] for x in dev[1]}
    steps = math.ceil(len(train[1]) / cfg['effective_batch_decisions']) * cfg['epochs']
    for epoch in range(latest, cfg['epochs']):
        began = time.monotonic()
        loss = train_epoch(head, optimizer, train, cfg, epoch, steps)
        value = {'epoch': epoch + 1, 'train_nll': loss, 'dev': metrics(predict(head, dev, ids)), 'seconds': time.monotonic() - began}
        epochs.append(value)
        atomic_save({'head': head.state_dict(), 'optimizer': optimizer.state_dict()}, out / f'epoch-{epoch + 1}.pt')
        write_json(out / f'epoch-{epoch + 1}.json', value)
        tracker.log({'fixed/epoch': epoch + 1, 'fixed/train_nll': loss, **{'fixed/dev_' + k: v for k, v in value['dev'].items()}})
        checkpoint()
        print(canonical({'event': 'fixed_cached_epoch', **value}), flush=True)
    chosen = best_epoch(epochs)
    location = out / f'epoch-{chosen}.pt' if chosen > latest else parent / 'fixed' / f'epoch-{chosen}.pt'
    head.load_state_dict(torch.load(location, weights_only=True, map_location='cuda')['head'])
    atomic_save(head.state_dict(), out / 'best-head.pt')
    write_json(out / 'best.json', {'epoch': chosen, 'selection': request['selection'], 'dev': next(x['dev'] for x in epochs if x['epoch'] == chosen)})
    write_json(out / 'training-complete.json', {'status': 'complete', 'epochs': cfg['epochs'], 'training_backbone_forwards': 0})
    checkpoint()
    # Load the frozen backbone only for native evaluation features that do not yet exist.
    backbone = processor = None
    frozen = []
    def features(name, rows, training):
        nonlocal backbone, processor, frozen
        original = parent / 'features' / name
        if (original / 'result.json').exists():
            return verified_cache(original)
        if backbone is None:
            from kernel_runtime import configure_optimized_runtime
            from resident_backbone import get_processor, get_model
            configure_optimized_runtime('both')
            processor = get_processor(cfg)
            verify_codes(processor.tokenizer, cfg['backbone_id'])
            backbone = get_model(cfg)
            frozen = [(p, p._version) for p in backbone.parameters()]
        prepared = [encode(processor.tokenizer, row, training, 'native', cfg['seed']) for row in rows]
        return extract(out / 'features' / name, prepared, backbone, processor.tokenizer, checkpoint, deadline)
    calibration = features('calibration-native', splits['calibration'], True)
    calibration_ids = {x['id']: x['option_ids'] for x in calibration[1]}
    fitted = fit_temperature(predict(head, calibration, calibration_ids))
    write_json(out / 'temperature.json', fitted)
    checkpoint()
    rows = load_panel()
    originals = {r['id']: r for r in rows}
    test = features('test-native', rows, False)
    test_ids = {r['id']: [key for key, _ in question_options(r['question'])] for r in rows}
    predictions = predict(head, test, test_ids)
    folder = out / 'test-native'
    folder.mkdir(exist_ok=True)
    (folder / 'logits.jsonl').write_text(''.join(canonical(x) + '\n' for x in predictions))
    for label, temp in [('raw', 1.), ('calibrated', fitted['temperature'])]:
        values = []
        for item in predictions:
            row = originals[item['id']]
            probabilities = (torch.tensor(item['logits'], dtype=torch.float64) / temp).softmax(-1).tolist()
            body = answer_body(row['question'], item['option_ids'], probabilities)
            values.append(record(row, dict(score(row, body), response=body)))
        (folder / (label + '-predictions.jsonl')).write_text(''.join(canonical(x) + '\n' for x in values))
    if not all(not p.requires_grad and p.grad is None and p._version == version for p, version in frozen):
        raise RuntimeError('Backbone changed')
    write_json(folder / 'config.json', {'panel_sha256': digest(ROOT / 'data/manifests/baseline-panel.json'),
                                       'temperature': fitted['temperature'], 'checkpoint_sha256': digest(out / 'best-head.pt')})
    write_json(folder / 'completion.json', {'status': 'complete', 'parameters_unchanged': True})
    from report_trained_joint import report
    summary = report(folder, label='Fixed residual MLP on v3; cached continuation')
    tracker.summary['test'] = summary
    tracker.finish()
    result = {'status': 'complete', 'request': request, 'best_epoch': chosen, 'test': summary, 'training_backbone_forwards': 0}
    write_json(out / 'result.json', result)
    seal(out)
    checkpoint()
    return result
