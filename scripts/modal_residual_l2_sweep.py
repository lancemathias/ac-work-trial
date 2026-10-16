"""Execute and export the E8 L2 sweep on existing frozen feature caches."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import modal
if not modal.is_local():
    sys.path.insert(0, '/project/scripts')
from modal_e6_zero_init import ROOT, ENV, image, results, secret

SWEEP='experiment-8-residual-l2-sweep-v2'
app=modal.App('ac-worktrial-e8-residual-l2-sweep')


def identity():
    paths=list((ROOT/'scripts').rglob('*.py'))+[ROOT/'configs/experiment-8-residual-l2-sweep.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


@app.function(image=image,gpu='A100-80GB',cpu=8,memory=16384,timeout=3600,retries=0,max_containers=1,volumes={'/results':results},secrets=[secret])
def fit(source):
    from sweep_residual_l2 import run
    if identity()!=source['sha256']:
        raise ValueError('Uploaded source mismatch')
    try:
        return run('/results',SWEEP,source,results.commit)
    finally:
        results.commit()


@app.function(image=image,cpu=4,memory=4096,timeout=1800,retries=0,volumes={'/results':results})
def bundle():
    import tarfile
    from training_data import digest
    out=Path('/results')/SWEEP
    assert json.loads((out/'result.json').read_text())['status']=='complete'
    for name,expected in json.loads((out/'checksums.json').read_text()).items():
        path=(out/name).resolve()
        assert path.is_relative_to(out.resolve()) and digest(path)==expected
    target=Path('/results')/(SWEEP+'.tar.gz')
    with tarfile.open(target,'w:gz',compresslevel=1) as stream:
        stream.add(out,arcname=SWEEP)
    results.commit()
    return {'archive':target.name,'bytes':target.stat().st_size,'sha256':digest(target)}


@app.local_entrypoint()
def main():
    from modal.config import _profile,config
    assert _profile=='ac-worktrial' and config.get('environment')==ENV
    source={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sha256':identity()}
    output=ROOT/'outputs/autoresearch'
    call=fit.spawn(source)
    (output/(SWEEP+'-launch.json')).write_text(json.dumps({'sweep_id':SWEEP,'function_call_id':call.object_id,'source':source,'profile':'ac-worktrial','workspace':'ac-worktrials','environment':ENV},indent=2)+'\n')
    print(json.dumps({'function_call_id':call.object_id,'sweep':SWEEP}),flush=True)
    value=call.get()
    (output/(SWEEP+'-result.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'winner':value['winner'],'test_accuracy':value['test']['macro_accuracy']}),flush=True)
    archive=bundle.remote()
    target=ROOT/'outputs/model-artifacts'/archive['archive']
    with target.open('wb') as stream:
        for chunk in results.read_file(archive['archive']):stream.write(chunk)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==archive['sha256']
    (output/(SWEEP+'-export.json')).write_text(json.dumps(dict(archive,path=str(target)),indent=2)+'\n')
    print(json.dumps({'export':str(target),'sha256':archive['sha256']}),flush=True)
