"""Audit every exported training file before releasing the source worker."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];NAMES=['ruletaker','aegis2','civilcomments','snli-multinli','massive','banking77']
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def verify():
 verified={};total=0
 for name in NAMES:
  meta=ROOT/'outputs/training'/('ab1-math-accuracy-seed20261014-'+name);receipt=json.loads((meta/'export-receipt.json').read_text())
  if receipt['status']!='verified':raise ValueError('Export not verified: '+name)
  source=Path(next(receipt[k] for k in ['local_export','source','artifact'] if k in receipt));checks=json.loads((meta/'checksums.json').read_text())
  for required in ['initial-head.safetensors','best-head.safetensors','last-training-state.pt','metrics.jsonl','summary.json','config.json','temperature.json']:
   if required not in checks:raise ValueError('Missing durable artifact: '+name+'/'+required)
  files={}
  for relative,expected in checks.items():
   path=(source/relative).resolve()
   if not path.is_relative_to(source.resolve()) or path.is_symlink():raise ValueError('Invalid export path')
   actual=digest(path)
   if actual!=expected:raise ValueError('Changed local export: '+name+'/'+relative)
   files[relative]={'sha256':actual,'bytes':path.stat().st_size};total+=path.stat().st_size
  verified[name]={'export':str(source),'tracked_manifest_sha256':digest(meta/'checksums.json'),'files':files}
 return {'status':'all_six_training_exports_reverified','total_bytes':total,'conditions':verified,'scope':'Every file in each tracked training export manifest, including initial/selected heads and terminal optimizer/training tensors. Does not certify pending BANKING77 test evaluation or permit benchmark-based fitting. No Modal resources cleaned.'}
if __name__=='__main__':
 value=verify();out=ROOT/'outputs/autoresearch/source-training-export-audit.json';out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':value['status'],'conditions':len(value['conditions']),'bytes':value['total_bytes']}))
