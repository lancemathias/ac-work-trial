"""Finish E8 locally exported reporting from saved checkpoint; no retraining or external artifact publication."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path
import modal
if not modal.is_local():
    sys.path.insert(0, "/project/scripts")
from modal_e6_zero_init import ROOT, ENV, RUN, image, results
app = modal.App('ac-worktrial-experiment-8-zero-output-init')

@app.function(image=image, cpu=8, memory=16384, timeout=1800, volumes={'/results': results})
def finish(run_id):
    sys.path.insert(0, '/project/scripts')
    import torch
    from cached_v31 import DATASET
    from finish_fixed_readout import verified_cache
    from train_residual_readout import predict
    from residual_code_readout import ResidualReadout
    from training_data import canonical, digest
    from head_probe_io import write_json, seal
    from run_baselines import load_panel, score, record
    from evaluate_trained_joint import answer_body
    from report_trained_joint import report
    torch.set_num_threads(8)
    out = Path('/results') / run_id
    request = json.loads((out / 'request.json').read_text())
    manifest = json.loads((Path('/results') / DATASET / 'manifest.json').read_text())
    test_path = Path(manifest['references']['test']['path'])
    assert digest(test_path / 'checksums.json') == manifest['references']['test']['checksums_sha256']
    cached = verified_cache(test_path)
    state = torch.load(out / 'best-head.pt', weights_only=True, map_location='cpu')
    head = ResidualReadout(state['code_weight'], state.get('code_bias'), 1024)
    head.load_state_dict(state)
    initial=torch.load(out/'initial-head.pt',weights_only=True,map_location='cpu')
    torch.testing.assert_close(head.code_weight,initial['code_weight'],rtol=0,atol=0)
    from train_joint import metrics
    train_cache = verified_cache(Path('/results') / DATASET / 'train')
    dev_cache = verified_cache(Path('/results') / DATASET / 'dev')
    curves = []
    for epoch in range(1, request['config']['epochs'] + 1):
        epoch_state = torch.load(out / f'epoch-{epoch}.pt', weights_only=True, map_location='cpu')['head']
        head.load_state_dict(epoch_state)
        row = json.loads((out / f'epoch-{epoch}.json').read_text())
        if 'train_eval' not in row:
            row['train_eval'] = metrics(predict(head, train_cache, {x['id']: x['option_ids'] for x in train_cache[1]}))
        row['dev_eval'] = metrics(predict(head, dev_cache, {x['id']: x['option_ids'] for x in dev_cache[1]}))
        curves.append(row)
        print(json.dumps({'epoch': epoch, 'online_train_nll': row['train_nll'], 'train_eval_nll': row['train_eval']['nll'], 'dev_eval_nll': row['dev_eval']['nll']}), flush=True)
    write_json(out / 'learning-curves.json', curves)
    head.load_state_dict(state)
    pred = predict(head, cached, {x['id']: x['option_ids'] for x in cached[1]})
    base = [json.loads(s) for s in (out / 'base-test-logits.jsonl').read_text().splitlines()]
    rows = {r['id']: r for r in load_panel()}
    temp = json.loads((out / 'temperature.json').read_text())['temperature']
    summaries = {}
    for label, predictions, temperature in [('test-native', pred, temp), ('base-test-native', base, 1.)]:
        folder = out / label
        folder.mkdir(exist_ok=True)
        (folder / 'logits.jsonl').write_text(''.join(canonical(x)+'\n' for x in predictions))
        for variant, t in [('raw',1.),('calibrated',temperature)]:
            scored=[]
            for item in predictions:
                row=rows[item['id']]
                body=answer_body(row['question'],item['option_ids'],(torch.tensor(item['logits'],dtype=torch.float64)/t).softmax(-1).tolist())
                scored.append(record(row,dict(score(row,body),response=body)))
            (folder / (variant+'-predictions.jsonl')).write_text(''.join(canonical(x)+'\n' for x in scored))
        write_json(folder/'config.json',{'panel_sha256':digest(ROOT/'data/manifests/baseline-panel.json'),'temperature':temperature,'checkpoint_sha256':digest(out/'best-head.pt')})
        write_json(folder/'completion.json',{'status':'complete','parameters_unchanged':True})
        summaries[label]=report(folder,label='Experiment 8' if label=='test-native' else 'Matched frozen code logits')
    info=json.loads((out/'wandb-run.json').read_text())
    value={'status':'complete','request':request,'experiment_number':8,'name':'Experiment 8: Zero-initialized residual output','best_epoch':json.loads((out/'best.json').read_text())['epoch'],'test':summaries['test-native'],'matched_base_test':summaries['base-test-native'],'wandb':info['url'],'optimizer_updates':len(curves) * 184,'backbone_loaded':False,'recovery':'Reporting only; selected checkpoint and completed training preserved'}
    write_json(out/'result.json',value)
    write_json(out/'experiment-name.json',{'number':8,'canonical_id':'experiment-8-zero-output-init-v31-seed20261014','legacy_storage_id':run_id})
    seal(out)
    archive=Path('/results')/(run_id+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:bundle.add(out,arcname=run_id)
    results.commit()
    return {'result':value,'archive':archive.name,'sha256':digest(archive),'bytes':archive.stat().st_size}

@app.local_entrypoint()
def main(run_id: str = RUN):
    from modal.config import _profile,config
    assert _profile=='ac-worktrial' and config.get('environment')==ENV
    value=finish.remote(run_id)
    output=ROOT/'outputs/autoresearch'
    (output/(run_id+'-result.json')).write_text(json.dumps(value,indent=2)+'\n')
    target=ROOT/'outputs/model-artifacts'/value['archive']
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as stream:
        for chunk in results.read_file(value['archive']):stream.write(chunk)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==value['sha256']
    (output/(run_id+'-export.json')).write_text(json.dumps({'path':str(target),'sha256':value['sha256'],'bytes':target.stat().st_size},indent=2)+'\n')
    print(json.dumps(value,indent=2))
