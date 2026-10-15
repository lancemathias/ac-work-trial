"""Idempotent retrospective import of verified benchmark results; no inference."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run():
 import wandb
 for line in (ROOT/'.env').read_text().splitlines():
  if '=' in line and not line.lstrip().startswith('#'):
   k,v=line.split('=',1)
   if k.strip()=='WANDB_API_KEY':os.environ[k.strip()]=v.strip().strip('\"\'')
 os.environ.update(WANDB_DISABLE_GIT='true',WANDB_DISABLE_CODE='true',WANDB_SILENT='true')
 ledger=json.loads((ROOT/'outputs/autoresearch/overnight-benchmark-scores.json').read_text())
 dest=ROOT/'outputs/handoff/wandb-imports.json';receipts=json.loads(dest.read_text()) if dest.exists() else []
 existing={r['label']:r for r in receipts}
 for label,scores in ledger['scores'].items():
  if label in existing:continue
  source=ledger['input_artifacts'][label];folder=None;files=[]
  for key,hashkey in [('path','sha256'),('prediction_path','prediction_sha256')]:
   if source.get(key):
    p=ROOT/source[key]
    if not p.exists():p=ROOT/'work/autoresearch'/source[key]
    if not p.exists() or digest(p)!=source[hashkey]:raise ValueError('Unverified source: '+label)
    folder=p.parent;files.append(p)
  if folder:
   for name in ['config.json','completion.json','results.md','temperature.json','best.json','artifact-manifest.json','checksums.json','wandb-run.json']:
    if (folder/name).is_file():files.append(folder/name)
  identity={'label':label,'panel':ledger['panel_sha256'],'source':source,'scores':scores}
  stable='hist'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
  historical=label not in ['Jev 1.13','PPLX v1','CLEF','CLEF Flash','E1','E2','E3','E4 fallback'] and not label.startswith('Random head')
  config=dict(import_kind='retrospective_existing_measurements',original_source=source,panel_sha256=ledger['panel_sha256'],aggregation='BBH subtask macro; equal 11-suite macro; misses wrong',autoresearch=historical,inference_performed_by_import=False)
  wb=wandb.init(entity='lmathias',project='ac-worktrial-decisions',id=stable,resume='allow',name='Evaluation / '+label,job_type='historical-evaluation',group='handoff-20261015',tags=['retrospective-import','autoresearch' if historical else 'baseline'],notes='Retrospective import of the existing fixed-panel evaluation. No training or inference was rerun. Original artifacts and hashes are attached; experiment rationale is in the chronological research document.',config=config,dir=str(ROOT/'work/deliverable-audit'))
  wb.log({'benchmark/'+k+'/accuracy':v for k,v in scores.items()})
  wb.summary.update({'verified_existing_measurement':True,'source':source,'status':'complete','benchmark_composite':scores['Composite']})
  artifact=wandb.Artifact(stable+'-evidence',type='benchmark-evidence',metadata=config)
  for p in dict.fromkeys(files):artifact.add_file(str(p),name=p.name)
  wb.log_artifact(artifact);url=wb.url;wb.finish()
  receipts.append({'label':label,'wandb_id':stable,'url':url,'source':source,'composite':scores['Composite'],'retrospective':True})
  dest.write_text(json.dumps(receipts,indent=2)+'\n');print(label,url,flush=True)
if __name__=='__main__':run()
