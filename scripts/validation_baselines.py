"""Released-model comparison on the fixed E2 validation set; no fitting."""
import argparse,asyncio,json,sys
from pathlib import Path
from run_baselines import sha,read_jsonl,write_json,run_jev,run_local

def load_validation(directory):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text());spec=m['splits']['dev']
    p=root/spec['points']['path']
    if sha(p)!=spec['points']['sha256']:raise ValueError('Validation split hash changed')
    rows=read_jsonl(p)
    if len(rows)!=262 or len({r['id'] for r in rows})!=262:raise ValueError('Unexpected E2 validation set')
    return rows,dict(kind='e2-validation-baselines',panel_sha256=sha(p),split='dev',expected=len(rows),split_manifest_sha256=sha(root/'manifest.json'),source_dataset_sha256=m['source_dataset_sha256'],calibration='Released probabilities; no fitting on validation labels')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--splits',required=True);p.add_argument('--out',required=True);p.add_argument('--key-file',required=True);a=p.parse_args()
    from dotenv import dotenv_values
    rows,metadata=load_validation(a.splits)
    asyncio.run(run_jev(a.out,rows=rows,metadata=metadata,api_key=dotenv_values(a.key_file).get('JEV_API_KEY')))
