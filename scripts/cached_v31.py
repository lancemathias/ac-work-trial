"""v3.1: native v3 views plus one existing shuffled view per choice training row."""
import copy
import json
import math
from pathlib import Path
import torch
from head_probe_io import prepare_run, write_json, seal
from finish_fixed_readout import verified_cache, best_epoch
from train_residual_readout import atomic_save, train_epoch, predict
from training_data import canonical, digest, load_splits
from residual_code_readout import ResidualReadout

PARENT = 'residual-readout1024-qwen38-v3-order-seed20261014'
FIXED = 'fixed-readout1024-qwen38-v3-cached-finish-wb-resume'
DATASET = 'readout-feature-dataset-v3.1-choice-permutation1'


def combine(native, shuffled, groups):
    features, items = native
    shuffled_features, shuffled_items = shuffled
    by_id = {x['id']: x for x in items}
    if set(by_id) != {x['id'] for x in shuffled_items}:
        raise ValueError('Shuffled view set differs from v3 train partition')
    selected, appended, identity = [], [], 0
    for index, item in enumerate(shuffled_items):
        original = by_id[item['id']]
        if item['type'] != original['type']:
            raise ValueError('Question type changed')
        if item['type'] != 'choice':
            continue
        if set(item['option_ids']) != set(original['option_ids']):
            raise ValueError('Semantic options changed')
        if item['option_ids'][item['target_index']] != original['option_ids'][original['target_index']]:
            raise ValueError('Permutation changed semantic target')
        identity += item['option_ids'] == original['option_ids']
        selected.append(index)
        view = copy.deepcopy(item)
        view.update(id=item['id'] + '::v3.1-shuffle-1', base_id=item['id'],
                    group_id=groups[item['id']], view='shuffle-1')
        appended.append(view)
    originals = [dict(x, base_id=x['id'], group_id=groups[x['id']], view='native') for x in items]
    merged = originals + appended
    if len({x['id'] for x in merged}) != len(merged):
        raise ValueError('View ID collision')
    return (torch.cat([features, shuffled_features[selected]]), merged), {
        'native_training_views': len(items), 'added_choice_views': len(appended),
        'training_views': len(merged), 'identity_permutations': identity,
        'note': 'Exactly the first seeded shuffled view is retained, including chance identity permutations; no additional resampling.'}


def save_cache(root, cached, identity):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    write_json(root / 'request.json', dict(identity, count=len(cached[1])))
    atomic_save({'features': cached[0], 'items': cached[1]}, root / 'shard-000000.pt')
    write_json(root / 'result.json', {'status': 'complete'})
    seal(root)


def build(root, source, checkpoint):
    results = Path(root)
    parent, fixed, out = results / PARENT, results / FIXED, results / DATASET
    cfg = json.loads((parent / 'config.json').read_text())['config']
    project = Path(__file__).resolve().parents[1]
    splits, split_manifest = load_splits(project / cfg['split_manifest'])
    request = {'version': 'v3.1', 'source': source, 'base_dataset_sha256': cfg['dataset']['sha256'],
               'base_request_sha256': digest(parent / 'request.json'), 'shuffle_epoch': 1, 'seed': cfg['seed'],
               'scope': 'native v3 training rows plus one existing shuffled view for choice training rows only; dev/calibration unchanged'}
    prior = prepare_run(out, request)
    if prior:
        return prior
    native = verified_cache(parent / 'features/train-fixed')
    shuffled = verified_cache(parent / 'features/train-shuffled-epoch-1')
    if {x['id'] for x in native[1]} != {x['id'] for x in splits['train']}:
        raise ValueError('Train cache differs from original split')
    groups = {x['id']: x['group_id'] for x in splits['train']}
    combined, stats = combine(native, shuffled, groups)
    save_cache(out / 'train', combined, request)
    (out / 'train-views.jsonl').write_text(''.join(canonical(x) + '\n' for x in combined[1]))
    references = {}
    for role, owner, name in [('dev', parent, 'dev-native'), ('calibration', fixed, 'calibration-native'), ('test', fixed, 'test-native')]:
        origin = owner / 'features' / name
        cached = verified_cache(origin)
        if role in splits and {x['id'] for x in cached[1]} != {x['id'] for x in splits[role]}:
            raise ValueError('Evaluation partition differs')
        if role == 'test':
            # Evaluation-only features are not part of the training dataset release.
            references[role] = {'path': str(origin), 'checksums_sha256': digest(origin / 'checksums.json')}
        else:
            save_cache(out / role, cached, dict(request, role=role, unchanged=True))
        references[role] = dict(references.get(role, {}), records=len(cached[1]), original_path=str(origin),
                                original_checksums_sha256=digest(origin / 'checksums.json'))
    write_json(out / 'manifest.json', {'version': 'v3.1', 'format': 'frozen-readout-feature-views-v1',
        'request': request, 'stats': stats, 'references': references, 'backbone_id': cfg['backbone_id'],
        'backbone_revision': cfg['backbone_revision'], 'codes_sha256': digest(project / 'configs/readout-mlp/codes-255.json'),
        'split_manifest_sha256': digest(project / cfg['split_manifest']),
        'sampling': 'Uniform over cached views: choice training rows have two views; noul rows have one.',
        'feature_extraction_repeated': False})
    result = {'status': 'complete', 'request': request, 'stats': stats, 'references': references}
    write_json(out / 'result.json', result)
    seal(out)
    checkpoint()
    return result


def train(results_root, run_id, source, checkpoint, zero_init_output=False):
    import wandb
    from train_joint import metrics, fit_temperature
    from run_baselines import load_panel, score, record
    from evaluate_trained_joint import answer_body
    from report_trained_joint import report
    root = Path(results_root)
    dataset = root / DATASET
    manifest = json.loads((dataset / 'manifest.json').read_text())
    cfg = json.loads((root / PARENT / 'config.json').read_text())['config']
    cfg = copy.deepcopy(cfg)
    if zero_init_output:
        cfg['head'].update(zero_init_output=True, initialization='E6 saved initialization with only down.weight and down.bias zeroed')
    request = {'zero_init_output': zero_init_output, 'dataset': DATASET, 'dataset_manifest_sha256': digest(dataset / 'manifest.json'), 'source': source,
               'config': cfg, 'selection': 'maximum native dev accuracy; earliest on ties',
               'feature_extraction': 'none; cached training/dev/calibration/test features only'}
    out = root / run_id
    prior = prepare_run(out, request)
    if prior:
        return prior
    write_json(out / 'config.json', request)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    train_cache, dev, calibration = [verified_cache(dataset / role) for role in ['train', 'dev', 'calibration']]
    initial = torch.load(root / PARENT / 'initial-head.pt', weights_only=True, map_location='cuda')
    head = ResidualReadout(initial['code_weight'], initial.get('code_bias'), 1024).cuda()
    head.load_state_dict(initial)
    if zero_init_output:
        with torch.no_grad():
            head.down.weight.zero_()
            head.down.bias.zero_()
    atomic_save(head.state_dict(), out / 'initial-head.pt')
    initial_rows = head.code_weight.clone()
    groups = [{'params': [p for p in head.parameters() if p.ndim >= 2], 'weight_decay': cfg['weight_decay']},
              {'params': [p for p in head.parameters() if p.ndim < 2], 'weight_decay': 0.}]
    optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], betas=tuple(cfg['adamw_betas']), eps=cfg['adamw_eps'])
    wb = wandb.init(project=cfg['wandb_project'], name=run_id, config=request, job_type='cached-v31-choice-permutation')
    write_json(out / 'wandb-run.json', {'id': wb.id, 'url': wb.url})
    ids = {x['id']: x['option_ids'] for x in dev[1]}
    steps = math.ceil(len(train_cache[1]) / cfg['effective_batch_decisions']) * cfg['epochs']
    initial_predictions = predict(head, dev, ids)
    write_json(out / 'initial-dev.json', metrics(initial_predictions))
    if zero_init_output:
        with torch.no_grad():
            sample = dev[0][:64].cuda().float()
            counts = torch.tensor([len(x['option_ids']) for x in dev[1][:64]], device='cuda')
            base = torch.nn.functional.linear(sample, head.code_weight, head.code_bias)
            valid = torch.arange(base.shape[-1], device='cuda')[None, :] < counts[:, None]
            torch.testing.assert_close(head(sample, counts), base.masked_fill(~valid, -torch.inf), rtol=0, atol=0)
        write_json(out / 'initial-base-parity.json', {'exact': True, 'rows': len(sample)})
    epochs = []
    for epoch in range(cfg['epochs']):
        loss = train_epoch(head, optimizer, train_cache, cfg, epoch, steps)
        values = {'epoch': epoch + 1, 'train_nll': loss, 'dev': metrics(predict(head, dev, ids))}
        epochs.append(values)
        write_json(out / f'epoch-{epoch + 1}.json', values)
        atomic_save({'head': head.state_dict(), 'optimizer': optimizer.state_dict()}, out / f'epoch-{epoch + 1}.pt')
        checkpoint()
        wb.log({'epoch': epoch + 1, 'train_nll': loss, **{'dev_' + k: v for k, v in values['dev'].items()}})
        print(canonical({'event': 'v31_cached_epoch', **values}), flush=True)
    chosen = best_epoch(epochs)
    head.load_state_dict(torch.load(out / f'epoch-{chosen}.pt', weights_only=True, map_location='cuda')['head'])
    atomic_save(head.state_dict(), out / 'best-head.pt')
    write_json(out / 'best.json', epochs[chosen - 1])
    fitted = fit_temperature(predict(head, calibration, {x['id']: x['option_ids'] for x in calibration[1]}))
    write_json(out / 'temperature.json', fitted)
    checkpoint()
    test_path = Path(manifest['references']['test']['path'])
    if digest(test_path / 'checksums.json') != manifest['references']['test']['checksums_sha256']:
        raise ValueError('Locked test feature reference changed')
    test = verified_cache(test_path)
    predictions = predict(head, test, {x['id']: x['option_ids'] for x in test[1]})
    if zero_init_output:
        base_head = copy.deepcopy(head)
        with torch.no_grad():
            base_head.down.weight.zero_()
            base_head.down.bias.zero_()
        base_predictions = predict(base_head, test, {x['id']: x['option_ids'] for x in test[1]})
        (out / 'base-test-logits.jsonl').write_text(''.join(canonical(x) + '\n' for x in base_predictions))
        del base_head
    originals = {r['id']: r for r in load_panel()}
    folder = out / 'test-native'
    folder.mkdir(exist_ok=True)
    (folder / 'logits.jsonl').write_text(''.join(canonical(x) + '\n' for x in predictions))
    for label, temp in [('raw', 1.), ('calibrated', fitted['temperature'])]:
        values = []
        for item in predictions:
            row = originals[item['id']]
            body = answer_body(row['question'], item['option_ids'], (torch.tensor(item['logits'], dtype=torch.float64) / temp).softmax(-1).tolist())
            values.append(record(row, dict(score(row, body), response=body)))
        (folder / (label + '-predictions.jsonl')).write_text(''.join(canonical(x) + '\n' for x in values))
    torch.testing.assert_close(head.code_weight, initial_rows, atol=0, rtol=0)
    project = Path(__file__).resolve().parents[1]
    write_json(folder / 'config.json', {'panel_sha256': digest(project / 'data/manifests/baseline-panel.json'),
                                       'temperature': fitted['temperature'], 'checkpoint_sha256': digest(out / 'best-head.pt')})
    write_json(folder / 'completion.json', {'status': 'complete', 'parameters_unchanged': True})
    summary = report(folder, label='v3.1 cached native plus one shuffled choice view')
    wb.summary['test'] = summary
    wb.finish()
    result = {'status': 'complete', 'request': request, 'best_epoch': chosen, 'test': summary, 'wandb': wb.url,
              'backbone_loaded': False, 'training_views': len(train_cache[1]), 'optimizer_updates': steps}
    write_json(out / 'result.json', result)
    seal(out)
    checkpoint()
    return result
