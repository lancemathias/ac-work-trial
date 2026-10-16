"""Apply the user-requested Experiment 9 name to the completed E8-derived sweep."""
import json
import sys
from pathlib import Path
import modal
if not modal.is_local():
    sys.path.insert(0,'/project/scripts')
from modal_e6_zero_init import ROOT, ENV, image, results, secret
LEGACY='experiment-8-residual-l2-sweep-v2'
GROUP='experiment-9-residual-l2-sweep'
app=modal.App('ac-worktrial-experiment-9-naming')

@app.function(image=image,cpu=2,memory=4096,timeout=600,volumes={'/results':results.read_only()},secrets=[secret])
def rename():
    import wandb
    result=json.loads((Path('/results')/LEGACY/'result.json').read_text())
    api=wandb.Api()
    renamed=[]
    for candidate in result['candidates']:
        run=api.run('lmathias/ac-worktrial-decisions/'+candidate['wandb'].rsplit('/',1)[-1])
        run.name=f"Experiment 9: Residual L2 (lambda={candidate['lambda']:g})"
        run.group=GROUP
        run.tags=list((set(run.tags)-{'experiment-8'})|{'experiment-9','l2-sweep','human-directed'})
        run.config.update({'experiment_number':9,'parent_experiment':8,'canonical_sweep_id':GROUP,'legacy_storage_id':LEGACY})
        run.update()
        verified=wandb.Api().run('lmathias/ac-worktrial-decisions/'+run.id)
        assert verified.name==run.name and verified.group==GROUP
        renamed.append({'lambda':candidate['lambda'],'id':run.id,'name':verified.name,'group':verified.group,'url':verified.url})
    return {'experiment_number':9,'parent_experiment':8,'canonical_sweep_id':GROUP,'legacy_storage_id':LEGACY,'group_url':'https://wandb.ai/lmathias/ac-worktrial-decisions/groups/'+GROUP,'runs':renamed}

@app.local_entrypoint()
def main():
    from modal.config import _profile,config
    assert _profile=='ac-worktrial' and config.get('environment')==ENV
    value=rename.remote()
    (ROOT/'outputs/autoresearch/experiment-9-wandb-naming.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
