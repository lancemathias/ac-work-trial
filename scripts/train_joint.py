"""E1: supervised joint head on frozen Qwen, then scalar temperature fitting."""
import argparse
import json
import math
import random
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from training_data import canonical, digest, load_splits
from vendor.clef_joint_schema import JointSchemaHead, collate_records, encode_record

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = '0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(canonical(value) + '\n')


def metrics(items, temperature=1.0):
    losses, briers, correct, confidences, ordinal_errors = [], [], [], [], []
    for item in items:
        logits = torch.tensor(item['logits'], dtype=torch.float64) / temperature
        probabilities = logits.softmax(-1)
        target = item['target_index']
        losses.append(float(-logits.log_softmax(-1)[target]))
        briers.append(float(((probabilities - F.one_hot(torch.tensor(target), len(logits))) ** 2).sum()))
        correct.append(int(int(probabilities.argmax()) == target))
        confidences.append(float(probabilities.max()))
        if item['type'] == 'score':
            ordinal_errors.append(abs(float((probabilities * torch.arange(len(logits))).sum()) - target))
    if not items:
        raise ValueError('Cannot evaluate an empty decision set')
    ece = 0.0
    for bucket in range(15):
        indices = [i for i, p in enumerate(confidences) if min(int(p * 15), 14) == bucket]
        if indices:
            ece += abs(sum(confidences[i] - correct[i] for i in indices)) / len(items)
    result = dict(decisions=len(items), nll=sum(losses)/len(items), brier=sum(briers)/len(items),
                  accuracy=sum(correct)/len(items), ece_15=ece)
    if ordinal_errors:
        result['score_mae'] = sum(ordinal_errors)/len(ordinal_errors)
    return result


def fit_temperature(items):
    """Fit one positive global T using labelled calibration data, never dev/test."""
    if not items:
        raise ValueError('Empty calibration set')
    # Bounded one-dimensional optimization includes T=1 as a fallback candidate.
    logits = [torch.tensor(i['logits'], dtype=torch.float64) for i in items]
    targets = [i['target_index'] for i in items]
    def objective(log_t):
        t = math.exp(log_t)
        return sum(float(-(z/t).log_softmax(-1)[y]) for z, y in zip(logits, targets, strict=True))/len(items)
    low, high = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1)/2
    left, right = high-ratio*(high-low), low+ratio*(high-low)
    fleft, fright = objective(left), objective(right)
    for _ in range(80):
        if fleft < fright:
            high, right, fright = right, left, fleft
            left = high-ratio*(high-low)
            fleft = objective(left)
        else:
            low, left, fleft = left, right, fright
            right = low+ratio*(high-low)
            fright = objective(right)
    candidates = [0.0, math.log(0.05), math.log(20.0), (low+high)/2]
    best = min(candidates, key=objective)
    return dict(temperature=math.exp(best), raw_nll=objective(0.0), calibrated_nll=objective(best),
                bounds=[0.05, 20.0], method='golden_section_log_temperature', decisions=len(items))


def slice_metrics(items, temperature=1.0):
    return {field: {key: metrics([item for item in items if item[field] == key], temperature)
                    for key in sorted({item[field] for item in items})}
            for field in ('source', 'type')}


def forward_frozen(backbone, head, batch, autocast=False):
    """no_grad, rather than inference_mode: head backward needs feature tensors."""
    backbone.eval()
    if hasattr(backbone, 'cached_features'):
        hidden = backbone.cached_features(batch).detach()
    else:
        text_model = backbone.model
        if hasattr(text_model, 'language_model'):
            text_model = text_model.language_model
        with torch.no_grad():
            hidden = text_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                use_cache=False, return_dict=True).last_hidden_state.detach()
    context = torch.autocast('cuda', dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        return head(hidden, batch['input_ids'], batch['attention_mask'], batch['records'],
                    backbone.get_output_embeddings().weight.detach())


def prepare(rows, tokenizer, max_tokens):
    prepared = []
    for row in rows:
        # Unlimited encoder budget followed by explicit rejection prevents silent truncation.
        encoded = encode_record(tokenizer, dict(row['request'], id=row['id']), max_length=10**9)
        if len(encoded.input_ids) > max_tokens:
            raise ValueError(f"{row['id']}: {len(encoded.input_ids)} tokens exceeds {max_tokens}; no truncation")
        prepared.append((row, encoded))
    return prepared


def decision_items(row, encoded, logits):
    if len(encoded.questions) != len(logits):
        raise ValueError('Head output does not match question count')
    items = []
    for question, values in zip(encoded.questions, logits, strict=True):
        if len(values) != len(question.option_ids) or not torch.isfinite(values).all():
            raise ValueError('Invalid/nonfinite logits')
        qid = question.question_id
        items.append(dict(id=row['id'], group_id=row['group_id'], source=row['source'],
                          question_id=qid, type=row['request']['questions'][qid]['type'],
                          option_ids=list(question.option_ids), logits=values.detach().float().cpu().tolist(),
                          target_index=question.option_ids.index(row['targets'][qid])))
    return items


def predict(backbone, head, prepared, pad, device, autocast):
    head.eval()
    items = []
    with torch.no_grad():
        for row, encoded in prepared:
            batch = collate_records([encoded], pad, device)
            logits = forward_frozen(backbone, head, batch, autocast)[0]
            items.extend(decision_items(row, encoded, logits))
    return items


def epoch_windows(prepared, seed, epoch, effective_batch):
    order = list(range(len(prepared)))
    random.Random(seed + epoch).shuffle(order)
    windows, window, count = [], [], 0
    for index in order:
        window.append(index)
        count += len(prepared[index][1].questions)
        if count >= effective_batch:
            windows.append(window)
            window, count = [], 0
    if window:
        windows.append(window)
    return windows


def train_head(backbone, head, prepared, pad, device, cfg, out, emit, checkpoint_callback=None, expected_initial_head_sha256=None):
    from safetensors.torch import save_file
    backbone.requires_grad_(False).eval()
    frozen = [(p, p._version) for p in backbone.parameters()]
    initial_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    save_file(initial_head, str(out/'initial-head.safetensors'))
    if expected_initial_head_sha256 and digest(out/'initial-head.safetensors') != expected_initial_head_sha256:
        raise ValueError('Initial head does not match original E1')
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    windows = [epoch_windows(prepared['train'], cfg['seed'], epoch, cfg['effective_batch_decisions'])
               for epoch in range(cfg['epochs'])]
    total_steps = sum(map(len, windows))
    warmup = max(1, math.ceil(total_steps * cfg['warmup_fraction']))
    autocast = device.type == 'cuda'
    step, best, stale = 0, math.inf, 0
    started = time.monotonic()
    emit(dict(event='untrained_dev', **metrics(predict(backbone, head, prepared['dev'], pad, device, autocast))))
    for epoch, batches in enumerate(windows):
        head.train()
        for window in batches:
            optimizer.zero_grad(set_to_none=True)
            loss_sum, count = 0.0, 0
            tick = time.monotonic()
            for index in window:
                row, encoded = prepared['train'][index]
                batch = collate_records([encoded], pad, device)
                logits = forward_frozen(backbone, head, batch, autocast)[0]
                losses = []
                for question, values in zip(encoded.questions, logits, strict=True):
                    target = question.option_ids.index(row['targets'][question.question_id])
                    losses.append(F.cross_entropy(values.float()[None], torch.tensor([target], device=device)))
                summed = torch.stack(losses).sum()
                if not torch.isfinite(summed):
                    raise ValueError('Nonfinite training loss')
                summed.backward()
                loss_sum += float(summed.detach())
                count += len(losses)
            for parameter in head.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(count)
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            scale = (step+1)/warmup if step < warmup else max(0.0, (total_steps-step)/max(1, total_steps-warmup))
            lr = cfg['learning_rate'] * scale
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            step += 1
            emit(dict(event='train_step', step=step, epoch=epoch+1, loss=loss_sum/count,
                      learning_rate=lr, gradient_norm=float(grad_norm), decisions=count,
                      decisions_per_second=count/(time.monotonic()-tick), elapsed_seconds=time.monotonic()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if autocast else 0,
                      record_ids=[prepared['train'][i][0]['id'] for i in window]))
        dev = metrics(predict(backbone, head, prepared['dev'], pad, device, autocast))
        emit(dict(event='dev_epoch', epoch=epoch+1, step=step, **dev))
        if dev['nll'] < best:
            best, stale = dev['nll'], 0
            temporary = out/'best-head.tmp.safetensors'
            save_file({k: v.detach().cpu().contiguous() for k,v in head.state_dict().items()}, str(temporary))
            temporary.replace(out/'best-head.safetensors')
            write_json(out/'best.json', dict(epoch=epoch+1, step=step, dev=dev))
        else:
            stale += 1
        # This checkpoint supports forensic inspection; automatic resume is not implemented.
        torch.save(dict(head=head.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1, step=step,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if autocast else [],
                        config=cfg), out/'last-training-state.pt')
        if checkpoint_callback:
            checkpoint_callback()
        if stale >= cfg['patience']:
            break
    if not all(not p.requires_grad and p.grad is None and p._version == version for p,version in frozen):
        raise RuntimeError('Frozen backbone changed or accumulated gradients')
    changed = any(not torch.equal(initial_head[k], value.detach().cpu()) for k,value in head.state_dict().items())
    return dict(steps=step, best_dev_nll=best, backbone_frozen=True, backbone_parameter_versions_unchanged=True,
                head_changed=changed, elapsed_seconds=time.monotonic()-started)


def run(config_path, manifest_path, output, checkpoint_callback=None, feature_cache=None, expected_initial_head_sha256=None):
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    cfg = json.loads(Path(config_path).read_text())
    validate_config(cfg)
    splits, manifest = load_splits(manifest_path)  # Validate before loading weights or allocating GPU.
    if cfg.get('dataset') and manifest.get('source_dataset_sha256') != cfg['dataset']['sha256']:
        raise ValueError('Split manifest does not match the pinned source dataset')
    if digest(ROOT/'scripts/vendor/clef_joint_schema.py') != UPSTREAM_SHA:
        raise ValueError('Vendored upstream source hash changed')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    wandb_run = None
    try:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg['seed'])
        device = torch.device(cfg['device'])
        torch.set_num_threads(cfg.get('cpu_threads', 8))
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.enable_cudnn_sdp(False)
        provenance = dict(config=cfg, manifest=manifest, config_sha256=digest(config_path),
                          manifest_sha256=digest(manifest_path), torch=torch.__version__,
                          training_point_schema_sha256=digest(ROOT/'configs/training-point.schema.json'),
                          transformers=__import__('transformers').__version__, head_source_sha256=UPSTREAM_SHA,
                          code_sha256={name: digest(ROOT/name) for name in ['scripts/train_joint.py', 'scripts/training_data.py']})
        write_json(out/'config.json', provenance)
        if cfg['wandb_mode'] != 'disabled':
            import wandb
            # Keep W&B's local files (including symlinks) out of the checksummed evidence directory.
            wandb_run = wandb.init(project=cfg['wandb_project'], name=out.name, config=provenance,
                                   mode=cfg['wandb_mode'], dir=tempfile.mkdtemp(prefix='wandb-'),
                                   notes=('Purpose: replicate E1 with all training choices held fixed, replacing repeated frozen Qwen forward passes with lossless cached hidden states. Compare training throughput and numerical agreement; same split, seed, initialization, optimizer, selection and calibration.' if feature_cache is not None else None))
        def emit(event):
            event = dict(event, timestamp_unix=time.time())
            append_json(out/'metrics.jsonl', event)
            print(canonical(event), flush=True)
            if wandb_run:
                # Let W&B use its monotonically increasing internal step across all phases.
                wandb_run.log({f"{event['event']}/{key}": value for key,value in event.items()
                               if key not in ('event', 'record_ids')})
        processor = AutoProcessor.from_pretrained(cfg['backbone_id'], revision=cfg['backbone_revision'], cache_dir=cfg['model_cache'])
        prepared = {role: prepare(rows, processor.tokenizer, cfg['max_input_tokens']) for role,rows in splits.items()}
        if feature_cache is None:
            backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
                cfg['backbone_id'], revision=cfg['backbone_revision'], cache_dir=cfg['model_cache'],
                dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
                attn_implementation='sdpa', device_map={'': str(device)})
        else:
            from frozen_feature_cache import FrozenFeatureBackbone
            cache_started = time.monotonic()
            backbone = FrozenFeatureBackbone(cfg=cfg, prepared=prepared, device=device, **feature_cache)
            provenance['feature_cache'] = dict(backbone.provenance, load_and_verify_seconds=time.monotonic()-cache_started)
            provenance['matched_initial_head_sha256'] = expected_initial_head_sha256
            provenance['cache_reader_sha256'] = digest(ROOT/'scripts/frozen_feature_cache.py')
            write_json(out/'config.json', provenance)
            if wandb_run:
                wandb_run.config.update({'feature_cache': provenance['feature_cache'],
                    'matched_initial_head_sha256': expected_initial_head_sha256})
            print(canonical(dict(event='cache_ready', **{'load_and_verify_seconds':time.monotonic()-cache_started,
                'cached_bytes':backbone.provenance['cached_bytes']})),flush=True)
        backbone.requires_grad_(False).eval()
        hidden_size = backbone.get_output_embeddings().weight.shape[1]
        torch.manual_seed(cfg['seed'])  # Keep head initialization independent of weight-loading RNG use.
        head = JointSchemaHead(hidden_size=hidden_size, **cfg['head']).float().to(device)
        result = train_head(backbone, head, prepared, processor.tokenizer.pad_token_id, device, cfg, out, emit, checkpoint_callback, expected_initial_head_sha256)
        head.load_state_dict(load_file(out/'best-head.safetensors', device=str(device)))
        calibration = predict(backbone, head, prepared['calibration'], processor.tokenizer.pad_token_id, device, device.type == 'cuda')
        fitted = fit_temperature(calibration)
        write_json(out/'temperature.json', fitted)
        dev = predict(backbone, head, prepared['dev'], processor.tokenizer.pad_token_id, device, device.type == 'cuda')
        checkpoint_hash = digest(out/'best-head.safetensors')
        for role, items in [('calibration', calibration), ('dev', dev)]:
            for item in items:
                raw = torch.tensor(item['logits'], dtype=torch.float64)
                append_json(out/f'{role}-traces.jsonl', dict(item, checkpoint_sha256=checkpoint_hash,
                    raw_probabilities=raw.softmax(-1).tolist(),
                    calibrated_probabilities=(raw/fitted['temperature']).softmax(-1).tolist()))
        if hasattr(backbone, 'verify_unchanged'):
            backbone.verify_unchanged()
            result['cached_features_unchanged'] = True
        report = dict(status='complete', **result, temperature=fitted,
                      dev_raw=metrics(dev), dev_calibrated=metrics(dev, fitted['temperature']),
                      dev_raw_slices=slice_metrics(dev), dev_calibrated_slices=slice_metrics(dev, fitted['temperature']),
                      calibration_raw=metrics(calibration), calibration_fitted=metrics(calibration, fitted['temperature']))
        write_json(out/'summary.json', report)
        emit(dict(event='final_dev_raw', **report['dev_raw']))
        emit(dict(event='final_dev_calibrated', **report['dev_calibrated']))
        if wandb_run:
            write_json(out/'wandb-run.json', dict(id=wandb_run.id, url=wandb_run.url, project=cfg['wandb_project']))
            wandb_run.summary.update(report)
            import wandb
            artifacts = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
            artifact = wandb.Artifact(out.name, type='e1-evidence', metadata=dict(
                artifact_sha256=artifacts, storage_path=str(out), modal_environment='worktrial-lance',
                modal_volume='ac-worktrial-training-results'))
            for name in ['config.json', 'best.json', 'temperature.json', 'summary.json',
                         'metrics.jsonl', 'dev-traces.jsonl', 'calibration-traces.jsonl']:
                artifact.add_file(str(out/name), name=name)
            wandb_run.log_artifact(artifact)
        return report
    except Exception as exc:
        write_json(out/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        if wandb_run:
            wandb_run.summary.update(dict(status='failed', error=type(exc).__name__, detail=str(exc)))
        raise
    finally:
        try:
            if wandb_run:
                wandb_run.finish()
        finally:
            files = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'checksums.json'}
            write_json(out/'checksums.json', files)
            if checkpoint_callback:
                checkpoint_callback()


def validate_config(cfg):
    if not isinstance(cfg.get('backbone_revision'), str) or len(cfg['backbone_revision']) != 40:
        raise ValueError('Pin backbone to a full commit revision')
    for key in ['epochs', 'effective_batch_decisions', 'max_input_tokens', 'patience']:
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid positive integer config: {key}')
    for key in ['learning_rate', 'gradient_clip']:
        if not math.isfinite(cfg.get(key, float('nan'))) or cfg[key] <= 0:
            raise ValueError(f'Invalid positive config: {key}')
    if not 0 <= cfg['warmup_fraction'] <= 1 or not math.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0:
        raise ValueError('Invalid warmup or weight decay')
    if cfg['wandb_mode'] not in ('online', 'offline', 'disabled'):
        raise ValueError('Invalid W&B mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.manifest, args.output), indent=2))
