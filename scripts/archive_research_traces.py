"""Losslessly version compact traces from checksum-verified local run exports."""
import gzip,hashlib,json
from pathlib import Path

def archive(source,destination):
 source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=True);items=[]
 for run in sorted(source.iterdir()):
  manifest=run/'checksums.json'
  if not manifest.is_file():continue
  hashes=json.loads(manifest.read_text())
  for name,expected in sorted(hashes.items()):
   path=(run/name).resolve()
   if not path.is_relative_to(run.resolve()):raise ValueError('Unsafe export path')
   base=path.name
   if not (base=='metrics.jsonl' or base.endswith(('-traces.jsonl','-predictions.jsonl','-logits.jsonl'))):continue
   raw=path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed source trace: '+str(path))
   target=destination/run.name/(name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():
    with target.open('wb') as f:
     with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=9) as z:z.write(raw)
   if hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()!=expected:raise ValueError('Archived trace differs: '+str(target))
   items.append({'run':run.name,'source_path':name,'archive':str(target.relative_to(destination)),'source_sha256':expected,'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'uncompressed_bytes':len(raw),'compressed_bytes':target.stat().st_size})
 index={'status':'verified','format':'Deterministic gzip JSONL, mtime0; source and decompressed SHA256 checked','files':items,'trace_count':len(items),'uncompressed_bytes':sum(x['uncompressed_bytes'] for x in items),'compressed_bytes':sum(x['compressed_bytes'] for x in items)}
 (destination/'index.json').write_text(json.dumps(index,indent=2)+'\n');return index
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();r=archive(a.source,a.destination);print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
