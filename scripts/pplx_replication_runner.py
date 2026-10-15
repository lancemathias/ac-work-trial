"""Instrumentation around unchanged upstream training; frozen mode changes only requires_grad."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys


def run(spec):
    import torch
    import autojev.train as upstream
    from autojev.model import DecisionModel, answer
    from autojev.types import Example
    retained = []
    versions = []

    class CapturedModel(DecisionModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if spec['mode'] == 'frozen':
                self.backbone.requires_grad_(False)
                versions.extend((p, p._version) for p in self.backbone.parameters())
            elif spec['mode'] != 'full':
                raise ValueError('Unknown replication mode')
            retained.append(self)

    upstream.DecisionModel = CapturedModel
    sys.argv = ['autojev-train', *spec['argv']]
    upstream.main()
    root = Path(spec['root'])
    summary = json.loads((root / 'run/summary.json').read_text())
    if not summary['complete']:
        raise RuntimeError('A partial training trajectory cannot be reported as a replication')
    model = retained.pop()
    if spec['mode'] == 'frozen' and not all(not p.requires_grad and p.grad is None and p._version == v for p, v in versions):
        raise RuntimeError('Frozen backbone invariant violated')
    # Save/test the actual terminal model as requested, separately from upstream selection.
    evaluations = [json.loads(line) for line in (root / 'run/evaluations.jsonl').read_text().splitlines()]
    model.temperature = evaluations[-1]['temperature']
    model.save(root / 'checkpoints/terminal', temperature=model.temperature, role='terminal', step=summary['steps'])
    sys.path.insert(0, '/project/scripts')
    from run_baselines import load_panel, score, record, canonical
    from head_probe_io import write_json, seal
    from training_data import digest
    from report_trained_joint import report
    rows = load_panel()
    def test(loaded, role, checkpoint_path):
        out = root / ('test-' + role)
        out.mkdir(exist_ok=False)
        scored = []
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            values = loaded.predict(batch, batch_size=4)
            for row, probabilities in zip(batch, values, strict=True):
                body = {'answers': {'q': answer(row['question'], probabilities)}}
                scored.append(record(row, dict(score(row, body), response=body)))
        (out / 'calibrated-predictions.jsonl').write_text(''.join(canonical(x) + '\n' for x in scored))
        write_json(out / 'config.json', {'panel_sha256': digest('/project/data/manifests/baseline-panel.json'),
            'temperature': loaded.temperature, 'checkpoint_sha256': digest(checkpoint_path / 'readout.safetensors')})
        write_json(out / 'completion.json', {'status': 'complete', 'parameters_unchanged': True})
        return report(out, label='PPLX v1 replication: ' + spec['mode'] + ' / ' + role)
    terminal = test(model, 'terminal', root / 'checkpoints/terminal')
    del model
    versions.clear()
    gc.collect()
    torch.cuda.empty_cache()
    selected = None
    if summary['best_step'] == summary['steps']:
        selected = dict(terminal, evaluation_reused='Upstream selected the terminal update; weights and fitted temperature are identical.')
    elif (root / 'checkpoints/selected').exists():
        model = DecisionModel(checkpoint=root / 'checkpoints/selected', train=False, cpu_threads=32)
        selected = test(model, 'upstream-selected', (root / 'checkpoints/selected').resolve())
    result = {'status': 'complete', 'mode': spec['mode'], 'terminal': terminal, 'upstream_selected': selected,
              'training_summary': summary, 'deviations': spec['deviations'], 'source': spec['source']}
    write_json(root / 'result.json', result)
    print(json.dumps({'replication_result': result}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    run(json.loads(Path(parser.parse_args().spec).read_text()))
