"""Evaluate the exact saved random MLP and a same-prompt frozen-only control on CPU."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path('/project/scripts').is_dir():
    sys.path.insert(0, '/project/scripts')
import modal
from modal_residual_readout import ROOT, ENV, image as base_image, results, secret
app = modal.App('ac-worktrial-random-residual-test')
image = base_image.add_local_file(ROOT / 'configs/random-residual-comparison.json', '/project/random-comparison.json')
PARENT = 'residual-readout1024-qwen38-v3-order-seed20261014'
FIXED = 'fixed-readout1024-qwen38-v3-cached-finish-wb-resume'
RUN = 'random-residual1024-qwen38-test-seed20261014'


@app.function(image=image, cpu=8, memory=8192, timeout=900, retries=0,
              volumes={'/results': results}, secrets=[secret])
def evaluate(source):
    import copy, torch, wandb
    from residual_code_readout import ResidualReadout
    from finish_fixed_readout import verified_cache
    from train_residual_readout import predict
    from train_joint import fit_temperature
    from head_probe_io import prepare_run, write_json, seal
    from training_data import digest, canonical
    from run_baselines import load_panel, score, record
    from evaluate_trained_joint import answer_body
    from report_trained_joint import report
    torch.set_num_threads(8)
    root, parent, cached = Path('/results') / RUN, Path('/results') / PARENT, Path('/results') / FIXED
    initial_path = parent / 'initial-head.pt'
    request = {'source': source, 'initial_head_sha256': digest(initial_path), 'seed': 20261014,
               'trained': False, 'backbone_loaded': False, 'calibration': 'same cached v3 calibration; positive T only'}
    prior = prepare_run(root, request)
    if prior:
        return prior
    calibration = verified_cache(cached / 'features/calibration-native')
    test = verified_cache(cached / 'features/test-native')
    initial = torch.load(initial_path, weights_only=True, map_location='cpu')
    head = ResidualReadout(initial['code_weight'], initial.get('code_bias'), 1024)
    head.load_state_dict(initial)
    head.requires_grad_(False).eval()
    pure = copy.deepcopy(head)
    with torch.no_grad():
        pure.down.weight.zero_()
        pure.down.bias.zero_()
    rows = {r['id']: r for r in load_panel()}
    wb = wandb.init(project='ac-worktrial-decisions', name=RUN, config=request, job_type='untrained-residual-test')
    write_json(root / 'wandb-run.json', {'id': wb.id, 'url': wb.url})
    summaries = {}
    for name, model in [('random_mlp', head), ('same_prompt_frozen_readout', pure)]:
        values = predict(model, calibration, {x['id']: x['option_ids'] for x in calibration[1]})
        fitted = fit_temperature(values)
        output = root / name
        output.mkdir(exist_ok=True)
        write_json(output / 'temperature.json', fitted)
        predictions = predict(model, test, {x['id']: x['option_ids'] for x in test[1]})
        (output / 'logits.jsonl').write_text(''.join(canonical(x) + '\n' for x in predictions))
        for label, temp in [('raw', 1.), ('calibrated', fitted['temperature'])]:
            scored = []
            for item in predictions:
                row = rows[item['id']]
                body = answer_body(row['question'], item['option_ids'], (torch.tensor(item['logits'], dtype=torch.float64) / temp).softmax(-1).tolist())
                scored.append(record(row, dict(score(row, body), response=body)))
            (output / (label + '-predictions.jsonl')).write_text(''.join(canonical(x) + '\n' for x in scored))
        write_json(output / 'config.json', {'panel_sha256': digest(ROOT / 'data/manifests/baseline-panel.json'),
                                           'temperature': fitted['temperature'], 'checkpoint_sha256': request['initial_head_sha256']})
        write_json(output / 'completion.json', {'status': 'complete', 'parameters_unchanged': True})
        summaries[name] = report(output, label=name)
        from report_prediction_dominance import summarize
        write_json(output / 'prediction-distributions.json', summarize(output / 'calibrated-predictions.jsonl'))
        results.commit()
    comparison = json.loads(Path('/project/random-comparison.json').read_text())
    comparison.update(summaries)
    wb.summary['comparison'] = comparison
    wb.finish()
    value = {'status': 'complete', 'request': request, 'wandb': wb.url, 'comparison': comparison}
    write_json(root / 'result.json', value)
    seal(root)
    results.commit()
    return value


@app.local_entrypoint()
def main():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    call = evaluate.spawn(source)
    receipt = {'run_id': RUN, 'function_call_id': call.object_id, 'cpu_only': True}
    (ROOT / 'outputs/autoresearch/random-residual-test-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
