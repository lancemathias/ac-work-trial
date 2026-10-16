"""User-authorized metrics/config sync only; never publish model checkpoints."""
import json
import subprocess
import sys
from pathlib import Path
import modal
if not modal.is_local():
    sys.path.insert(0,'/project/scripts')
from modal_e6_zero_init import ROOT, ENV, image, results, secret
RUN='experiment-8-zero-output-init-v31-10epochs-seed20261014'
app=modal.App('ac-worktrial-experiment-8-metrics-sync')

@app.function(image=image,cpu=2,memory=4096,timeout=600,volumes={'/results':results.read_only()},secrets=[secret])
def sync():
    import os
    import wandb
    out=Path('/results')/RUN
    info=json.loads((out/'wandb-run.json').read_text())
    request=json.loads((out/'request.json').read_text())
    wb=wandb.init(entity='lmathias',project='ac-worktrial-decisions',id=info['id'],resume='allow',mode='online',dir='/tmp',
                  name='Experiment 8: Zero-initialized residual output (10 epochs)',config=request,
                  tags=['experiment-8','human-directed','zero-output-init','10-epochs'])
    curves=json.loads((out/'learning-curves.json').read_text())
    for row in curves:
        values={'epoch':row['epoch'],'train_nll':row['train_nll'],'train_loss':row['train_nll']}
        values.update({'train_eval_'+k:v for k,v in row['train_eval'].items()})
        values.update({'dev_'+k:v for k,v in row['dev'].items()})
        wb.log(values,step=row['epoch'])
    wb.config.update({'experiment_number':8,'parent_experiment':6,'epochs':10,'optimizer_updates':1840},allow_val_change=True)
    best=json.loads((out/'best.json').read_text())
    wb.summary['best_epoch']=best['epoch']
    wb.summary['best_dev_accuracy']=best['dev']['accuracy']
    for name,folder in [('test','test-native'),('matched_base_test','base-test-native')]:
        wb.summary[name]=json.loads((out/folder/'metrics.json').read_text())
    url=wb.url
    name=wb.name
    wb.finish()
    return {'url':url,'id':info['id'],'name':name,'payload':'metrics and configuration only; no model checkpoints'}

@app.local_entrypoint()
def main():
    from modal.config import _profile,config
    assert _profile=='ac-worktrial' and config.get('environment')==ENV
    value=sync.remote()
    (ROOT/'outputs/autoresearch/experiment-8-wandb-sync.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
