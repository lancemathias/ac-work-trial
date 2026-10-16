"""Matched E8 centered-residual L2 sweep. Test access follows locked dev selection."""
import copy
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from residual_code_readout import ResidualReadout
from finish_fixed_readout import verified_cache, best_epoch
from cached_v31 import DATASET, PARENT
from train_residual_readout import predict, atomic_save
from head_probe_io import write_json, prepare_run, seal
from training_data import digest, canonical

PARENT_RUN = 'experiment-8-zero-output-init-v31-10epochs-seed20261014'
ROOT = Path(__file__).resolve().parents[1]


def residual_penalty(delta, counts):
    valid = torch.arange(delta.shape[-1], device=delta.device)[None, :] < counts[:, None]
    active = delta.masked_fill(~valid, 0.)
    mean = active.sum(-1, keepdim=True) / counts[:, None]
    centered = (active - mean).masked_fill(~valid, 0.)
    return (centered.square().sum(-1) / counts).mean()


def contracts():
    counts = torch.tensor([2, 3])
    x = torch.tensor([[1., 3., 999.], [0., 0., 0.]], requires_grad=True)
    p = residual_penalty(x, counts)
    torch.testing.assert_close(p, torch.tensor(.5))
    shifted = x + torch.tensor([[17.], [-8.]])
    torch.testing.assert_close(p, residual_penalty(shifted, counts))
    p.backward()
    assert x.grad[0, 2] == 0 and torch.isfinite(x.grad).all()
    zero = torch.zeros(2, 4, requires_grad=True)
    residual_penalty(zero, torch.tensor([2, 4])).backward()
    assert torch.isfinite(zero.grad).all() and zero.grad.abs().sum() == 0
    return {'valid_option_masking': True, 'shift_invariance': True, 'finite_zero_gradient': True}


def epoch(head, optimizer, cached, cfg, epoch_index, total_steps, strength):
    features, items = cached
    generator = torch.Generator().manual_seed(cfg['seed'] + epoch_index)
    indices = torch.randperm(len(items), generator=generator)
    steps = math.ceil(len(items) / cfg['effective_batch_decisions'])
    warmup = max(1, int(total_steps * cfg['warmup_fraction']))
    sums = dict(nll=0., residual_l2=0., objective=0.)
    head.train()
    for batch, start in enumerate(range(0, len(items), cfg['effective_batch_decisions'])):
        take = indices[start:start + cfg['effective_batch_decisions']]
        group = [items[i] for i in take.tolist()]
        step = epoch_index * steps + batch
        scale = (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for params in optimizer.param_groups:
            params['lr'] = cfg['learning_rate'] * scale
        hidden = features[take].cuda().float()
        counts = torch.tensor([len(x['option_ids']) for x in group], device='cuda')
        targets = torch.tensor([x['target_index'] for x in group], device='cuda')
        optimizer.zero_grad(set_to_none=True)
        delta = head.down(F.gelu(head.up(head.norm(hidden))))
        logits = F.linear(hidden, head.code_weight, head.code_bias) + delta
        valid = torch.arange(logits.shape[-1], device='cuda')[None, :] < counts[:, None]
        ce = F.cross_entropy(logits.masked_fill(~valid, -torch.inf), targets)
        penalty = residual_penalty(delta, counts)
        loss = ce + strength * penalty
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        for name, value in [('nll', ce), ('residual_l2', penalty), ('objective', loss)]:
            sums[name] += float(value.detach()) * len(group)
    return {k: v / len(items) for k, v in sums.items()}


@torch.no_grad()
def evaluate(head, cached):
    from train_joint import metrics
    features, items = cached
    values = metrics(predict(head, cached, {x['id']: x['option_ids'] for x in items}))
    penalties = []
    head.eval()
    for start in range(0, len(items), 64):
        group = items[start:start + 64]
        x = features[start:start + len(group)].cuda().float()
        delta = head.down(F.gelu(head.up(head.norm(x))))
        counts = torch.tensor([len(x['option_ids']) for x in group], device='cuda')
        penalties.append(float(residual_penalty(delta, counts)) * len(group))
    values['residual_l2'] = sum(penalties) / len(items)
    values['residual_rms'] = math.sqrt(values['residual_l2'])
    return values


def run(results_root, sweep_id, source, checkpoint):
    root = Path(results_root)
    spec = json.loads((ROOT / 'configs/experiment-8-residual-l2-sweep.json').read_text())
    parent = root / PARENT_RUN
    parent_request = json.loads((parent / 'request.json').read_text())
    cfg = copy.deepcopy(parent_request['config'])
    assert cfg['epochs'] == spec['epochs'] == 10 and cfg['seed'] == spec['seed']
    request = {'spec': spec, 'source': source, 'parent_request_sha256': digest(parent / 'request.json'),
               'initial_state_sha256': digest(parent / 'initial-head.pt'), 'config': cfg,
               'dataset_manifest_sha256': digest(root / DATASET / 'manifest.json')}
    out = root / sweep_id
    prior = prepare_run(out, request)
    if prior:
        return prior
    write_json(out / 'contracts.json', contracts())
    write_json(out / 'config.json', request)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    train, dev = [verified_cache(root / DATASET / role) for role in ['train', 'dev']]
    initial = torch.load(parent / 'initial-head.pt', weights_only=True, map_location='cuda')
    assert initial['down.weight'].count_nonzero() == 0 and initial['down.bias'].count_nonzero() == 0
    steps = math.ceil(len(train[1]) / cfg['effective_batch_decisions']) * cfg['epochs']
    candidates = []
    import wandb
    for strength in spec['lambdas']:
        folder = out / ('lambda-' + format(strength, 'g'))
        folder.mkdir(exist_ok=True)
        head = ResidualReadout(initial['code_weight'], initial.get('code_bias'), 1024).cuda()
        head.load_state_dict(initial)
        groups = [{'params': [p for p in head.parameters() if p.ndim >= 2], 'weight_decay': cfg['weight_decay']},
                  {'params': [p for p in head.parameters() if p.ndim < 2], 'weight_decay': 0.}]
        optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], betas=tuple(cfg['adamw_betas']), eps=cfg['adamw_eps'])
        wb = wandb.init(entity='lmathias', project=cfg['wandb_project'], name=f'E8 residual L2 sweep: lambda={strength:g}', group=sweep_id, job_type='centered-residual-l2-sweep', config=dict(request, residual_l2_lambda=strength), tags=['experiment-8', 'l2-sweep', 'human-directed'])
        write_json(folder / 'wandb-run.json', {'id': wb.id, 'url': wb.url})
        records = []
        best = None
        for e in range(cfg['epochs']):
            online = epoch(head, optimizer, train, cfg, e, steps, strength)
            record = {'lambda': strength, 'epoch': e + 1, 'online_train': online, 'train_eval': evaluate(head, train), 'dev': evaluate(head, dev)}
            records.append(record)
            wb.log({'epoch': e+1, 'train_nll': online['nll'], 'train_loss': online['nll'], 'train_residual_l2': online['residual_l2'], 'train_objective': online['objective'], **{'train_eval_'+k:v for k,v in record['train_eval'].items()}, **{'dev_'+k:v for k,v in record['dev'].items()}})
            write_json(folder / f'epoch-{e+1}.json', record)
            atomic_save({'head': head.state_dict(), 'optimizer': optimizer.state_dict()}, folder / f'epoch-{e+1}.pt')
            if best is None or record['dev']['accuracy'] > best['dev']['accuracy']:
                best = record
                atomic_save(head.state_dict(), folder / 'best-head.pt')
                write_json(folder / 'best.json', best)
            print(canonical({'event': 'l2_epoch', 'lambda': strength, 'epoch': e+1,
                             'train_nll': record['train_eval']['nll'], 'dev_nll': record['dev']['nll'],
                             'dev_accuracy': record['dev']['accuracy'], 'dev_residual_rms': record['dev']['residual_rms']}), flush=True)
            checkpoint()
        torch.testing.assert_close(head.code_weight, initial['code_weight'], rtol=0, atol=0)
        if strength == 0:
            reference = [json.loads((parent / f'epoch-{e+1}.json').read_text()) for e in range(10)]
            parity = {'max_dev_nll_difference': max(abs(a['dev']['nll']-b['dev']['nll']) for a,b in zip(records, reference)),
                      'all_dev_accuracies_equal': all(a['dev']['accuracy']==b['dev']['accuracy'] for a,b in zip(records, reference))}
            write_json(folder / 'e8-control-parity.json', parity)
            assert parity['all_dev_accuracies_equal'] and parity['max_dev_nll_difference'] < 1e-5, parity
        wb.summary['best_epoch'] = best['epoch']
        wb.summary['best_dev'] = best['dev']
        wb.finish()
        write_json(folder / 'learning-curves.json', records)
        candidates.append({'lambda': strength, 'best_epoch': best['epoch'], 'dev': best['dev'], 'folder': folder.name, 'wandb': wb.url})
        write_json(out / 'candidates.json', candidates)
        del head, optimizer
    winner = max(candidates, key=lambda c: (c['dev']['accuracy'], -c['dev']['nll'], -c['lambda']))
    # Freeze the sweep choice before any test feature or label access.
    write_json(out / 'selection.json', {'winner': winner, 'candidates': candidates, 'test_accessed': False})
    checkpoint()
    folder = out / winner['folder']
    head = ResidualReadout(initial['code_weight'], initial.get('code_bias'), 1024).cuda()
    head.load_state_dict(torch.load(folder / 'best-head.pt', weights_only=True, map_location='cuda'))
    from train_joint import fit_temperature
    calibration = verified_cache(root / DATASET / 'calibration')
    fitted = fit_temperature(predict(head, calibration, {x['id']: x['option_ids'] for x in calibration[1]}))
    write_json(folder / 'temperature.json', fitted)
    manifest = json.loads((root / DATASET / 'manifest.json').read_text())
    test_path = Path(manifest['references']['test']['path'])
    assert digest(test_path / 'checksums.json') == manifest['references']['test']['checksums_sha256']
    test = verified_cache(test_path)
    predictions = predict(head, test, {x['id']: x['option_ids'] for x in test[1]})
    from run_baselines import load_panel, score, record as score_record
    from evaluate_trained_joint import answer_body
    from report_trained_joint import report
    originals = {r['id']: r for r in load_panel()}
    test_out = folder / 'test-native'
    test_out.mkdir(exist_ok=True)
    (test_out / 'logits.jsonl').write_text(''.join(canonical(x)+'\n' for x in predictions))
    for name, temperature in [('raw',1.), ('calibrated',fitted['temperature'])]:
        scored=[]
        for item in predictions:
            row=originals[item['id']]
            body=answer_body(row['question'],item['option_ids'],(torch.tensor(item['logits'],dtype=torch.float64)/temperature).softmax(-1).tolist())
            scored.append(score_record(row,dict(score(row,body),response=body)))
        (test_out/(name+'-predictions.jsonl')).write_text(''.join(canonical(x)+'\n' for x in scored))
    write_json(test_out/'config.json',{'panel_sha256':digest(ROOT/'data/manifests/baseline-panel.json'),'temperature':fitted['temperature'],'checkpoint_sha256':digest(folder/'best-head.pt')})
    write_json(test_out/'completion.json',{'status':'complete','parameters_unchanged':True})
    test_metrics=report(test_out,label='E8 centered-residual L2 sweep; dev-selected winner')
    winner_info = json.loads((folder / 'wandb-run.json').read_text())
    winner_wb = wandb.init(entity='lmathias', project=cfg['wandb_project'], id=winner_info['id'], resume='must')
    winner_wb.summary['sweep_selected'] = True
    winner_wb.summary['test'] = test_metrics
    winner_wb.finish()
    result={'status':'complete','request':request,'winner':winner,'candidates':candidates,'test':test_metrics,
            'backbone_loaded':False,'updates_per_lambda':steps,'total_updates':steps*len(spec['lambdas'])}
    write_json(out/'result.json',result)
    seal(out)
    checkpoint()
    return result
