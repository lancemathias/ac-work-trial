"""Restore committed compact evaluation traces; refuse different existing bytes."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def restore():
 base=ROOT/'outputs/traces';index=json.loads((base/'index.json').read_text());count=0
 for item in index['files']:
  source=(base/item['archive']).resolve()
  if not source.is_relative_to(base.resolve()):raise ValueError('Unsafe archive path')
  compressed=source.read_bytes()
  if hashlib.sha256(compressed).hexdigest()!=item['archive_sha256']:raise ValueError('Changed archive')
  raw=gzip.decompress(compressed)
  if hashlib.sha256(raw).hexdigest()!=item['source_sha256']:raise ValueError('Changed trace')
  dest=(ROOT/'outputs/model-artifacts'/item['run']/item['source_path']).resolve()
  if not dest.is_relative_to((ROOT/'outputs/model-artifacts').resolve()):raise ValueError('Unsafe destination')
  if dest.exists() and dest.read_bytes()!=raw:raise ValueError('Different existing evidence: '+str(dest))
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);count+=1
 # Restore the committed manifests needed to verify the decompressed evidence.
 for parent in [ROOT/'outputs/test',ROOT/'outputs/training',ROOT/'outputs/handoff/export-metadata']:
  for folder in parent.iterdir():
   target=ROOT/'outputs/model-artifacts'/folder.name
   if not target.exists():continue
   for source in folder.glob('*.json'):
    dest=target/source.name
    if source.exists():
     data=source.read_bytes()
     if dest.exists() and dest.read_bytes()!=data:
      if parent.name=='export-metadata':continue  # Full-attempt metadata can differ from flattened training metadata.
      raise ValueError('Different existing manifest: '+str(dest))
     dest.write_bytes(data)
 print('Verified/restored',count,'trace files; no inference performed')
if __name__=='__main__':restore()
