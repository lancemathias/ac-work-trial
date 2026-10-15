"""E2: normalized v2 cache, four-record microbatches, five full epochs."""
import json,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
ENVIRONMENT='worktrial-lance'
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
       .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs'))
app=modal.App('ac-worktrial-e2-v2-cached-training')
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENVIRONMENT)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENVIRONMENT)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENVIRONMENT)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENVIRONMENT,required_keys=['WANDB_API_KEY'])

@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=12*3600,max_containers=1,
 volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def train(run_id,commit):
    import sys
    sys.path.insert(0,'/project/scripts')
    from train_joint import run
    from training_data import digest,canonical,read_jsonl,validate_run_id
    from benchmark_training_data import adapt_training_point
    from prepare_training_splits import assign_roles
    from jsonschema import Draft202012Validator
    validate_run_id(run_id)
    config=Path('/project/configs/e2-qwen35-9b-v2-cache.json');cfg=json.loads(config.read_text())
    source=Path('/data'+cfg['dataset']['path'])
    if digest(source)!=cfg['dataset']['sha256']:raise ValueError('v2 source changed')
    schema=Path('/project/configs/normalized-training.schema.json')
    if digest(schema)!=cfg['dataset']['schema_sha256']:raise ValueError('v2 schema changed')
    points=read_jsonl(source);validator=Draft202012Validator(json.loads(schema.read_text()))
    adapted=[adapt_training_point(p,validator) for p in points]
    # Reuse E1's connected-group/exact-state hash partition rule and seed.
    roles=assign_roles([dict(id=r['id'],family_id=r['group_id'],model_payload={'input':canonical({'state':r['request']['state']})}) for r in adapted],cfg['seed'])
    splitdir=Path('/results')/(run_id+'-splits');splitdir.mkdir(exist_ok=False)
    report=dict(approved=True,dataset_sha256=digest(source),basis='User explicitly authorized E2 on v2 cached dataset; authorization attestation, not an independent contamination audit.',commit=commit)
    (splitdir/'hygiene-report.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=dict(version=1,format='normalized-benchmark-v1',source_dataset_sha256=digest(source),training_point_schema_sha256=digest(schema),split_seed=cfg['seed'],split_rule='SHA256([seed, connected family/state group]); 80/10/10 hash thresholds',hygiene=dict(approved=True,report=dict(path='hygiene-report.json',sha256=digest(splitdir/'hygiene-report.json'))),splits={})
    for role in ['train','dev','calibration']:
        rows=[p for p in points if roles[p['id']]==role]
        path=splitdir/(role+'.jsonl');path.write_text(''.join(canonical(p)+'\n' for p in rows))
        manifest['splits'][role]=dict(role=role,evaluation_only=False,allowed_sources=sorted({p['suite'] for p in rows}),count=len(rows),points=dict(path=path.name,sha256=digest(path)))
    mp=splitdir/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');results.commit()
    expected=json.loads(Path('/project/configs/e2-cache-source.json').read_text())['config']['config']
    try:
        return run(config,mp,Path('/results')/run_id,checkpoint_callback=results.commit,
          expected_initial_head_sha256='2fe908f8ac0f25f81238b4f4c59b23377dc27665532741483b043a1b99bcf957',
          feature_cache=dict(directory='/data/hidden-cache/qwen35-9b-v2-batch4-338a3584e896',expected_manifest_sha256='b461903e3990adc1bfbbf649b7e5c658f617dee3e76be4e61baead4fbce8d293',expected_adapter_sha256=digest('/project/scripts/benchmark_training_data.py'),expected_source_config=expected,expected_extraction_batch_size=4))
    finally:results.commit();models.commit()

@app.local_entrypoint()
def main(run_id:str):
    from modal.config import config,_profile
    if _profile!='ac-worktrial' or config.get('environment')!=ENVIRONMENT:raise ValueError('Use ac-worktrial profile and worktrial-lance environment')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    print(json.dumps(train.remote(run_id,commit),indent=2))
