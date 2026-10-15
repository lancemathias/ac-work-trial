"""Apply the existing grouped split rule to the user-authorized v3 release."""
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from training_data import canonical,digest,read_jsonl
from benchmark_training_data import adapt_training_point
from prepare_training_splits import assign_roles

def prepare(source,destination,cfg):
    source=Path(source);out=Path(destination)
    if digest(source)!=cfg['dataset']['sha256']:raise ValueError('v3 source hash mismatch')
    schema=Path(__file__).resolve().parents[1]/'configs/normalized-training.schema.json'
    validator=Draft202012Validator(json.loads(schema.read_text()))
    points=read_jsonl(source);rows=[adapt_training_point(p,validator) for p in points]
    roles=assign_roles([dict(id=r['id'],family_id=r['group_id'],model_payload={'input':canonical({'state':r['request']['state']})}) for r in rows],cfg['seed'])
    out.mkdir(parents=True,exist_ok=False)
    audit=dict(approved=True,dataset_sha256=digest(source),basis='User authorized E3 on the v3 cached dataset; authorization attestation, not an independent data audit.')
    (out/'hygiene-report.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest=dict(version=1,format='normalized-benchmark-v1',source_dataset_sha256=digest(source),training_point_schema_sha256=digest(schema),split_seed=cfg['seed'],split_rule='SHA256([seed, connected family/state group]); 80/10/10 hash thresholds',hygiene=dict(approved=True,report=dict(path='hygiene-report.json',sha256=digest(out/'hygiene-report.json'))),splits={})
    for role in ['train','dev','calibration']:
        selected=[p for p in points if roles[p['id']]==role];path=out/(role+'.jsonl')
        path.write_text(''.join(canonical(p)+'\n' for p in selected))
        manifest['splits'][role]=dict(role=role,evaluation_only=False,allowed_sources=sorted({p['suite'] for p in selected}),count=len(selected),points=dict(path=path.name,sha256=digest(path)))
    mp=out/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    return mp
