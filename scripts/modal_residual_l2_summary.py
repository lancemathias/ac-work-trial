"""Publish selected-epoch sweep summaries and export small local metric tables."""
import json
import sys
from pathlib import Path
import modal
if not modal.is_local():
    sys.path.insert(0, '/project/scripts')
from modal_e6_zero_init import ROOT, ENV, image, results, secret
SWEEP='experiment-8-residual-l2-sweep-v2'
app=modal.App('ac-worktrial-e8-l2-summary')

@app.function(image=image,cpu=2,memory=4096,timeout=600,volumes={'/results':results.read_only()},secrets=[secret])
def summarize():
    import wandb
    out=Path('/results')/SWEEP
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='complete'
    api=wandb.Api()
    curves={}
    for candidate in result['candidates']:
        folder=out/candidate['folder']
        curves[candidate['folder']]=json.loads((folder/'learning-curves.json').read_text())
        info=json.loads((folder/'wandb-run.json').read_text())
        remote=api.run('lmathias/ac-worktrial-decisions/'+info['id'])
        remote.summary.update({'best_epoch':candidate['best_epoch'],'best_dev_accuracy':candidate['dev']['accuracy'],
            'best_dev_nll':candidate['dev']['nll'],'best_dev_residual_rms':candidate['dev']['residual_rms'],
            'residual_l2_lambda':candidate['lambda'],'sweep_selected':candidate['lambda']==result['winner']['lambda']})
    return {'result':result,'curves':curves,'contracts':json.loads((out/'contracts.json').read_text()),
            'control_parity':json.loads((out/'lambda-0/e8-control-parity.json').read_text())}

@app.local_entrypoint()
def main():
    from modal.config import _profile,config
    assert _profile=='ac-worktrial' and config.get('environment')==ENV
    value=summarize.remote()
    out=ROOT/'outputs/training'/SWEEP
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(value['result'],indent=2)+'\n')
    for folder,curves in value['curves'].items():
        (out/(folder+'-curves.json')).write_text(json.dumps(curves,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({k:value[k] for k in ['contracts','control_parity']},indent=2)+'\n')
    import csv
    with (out/'comparison.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['lambda','best_epoch','dev_accuracy','dev_nll','dev_residual_rms','selected','wandb'])
        for c in value['result']['candidates']:
            writer.writerow([c['lambda'],c['best_epoch'],c['dev']['accuracy'],c['dev']['nll'],c['dev']['residual_rms'],c['lambda']==value['result']['winner']['lambda'],c['wandb']])
    print(json.dumps({'winner':value['result']['winner'],'test_accuracy':value['result']['test']['macro_accuracy']},indent=2))
