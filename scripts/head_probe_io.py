"""Preemption-safe diagnostic identity and immutable completion checks."""
import hashlib,json,uuid
from pathlib import Path

def write_json(path,value):
 path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def seal(root):
 root=Path(root);manifest=root/'checksums.json'
 if manifest.exists():
  for name,expected in json.loads(manifest.read_text()).items():
   path=(root/name).resolve()
   if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Previously sealed probe artifact changed')
 scratch=root/'checksums.tmp'
 if scratch.exists():scratch.rename(root/('preempted-checksums-'+uuid.uuid4().hex+'.tmp'))
 hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='checksums.json'}
 write_json(manifest,hashes)

def prepare_run(root,request):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);identity=root/'request.json'
 if identity.exists():
  if json.loads(identity.read_text())!=request:raise ValueError('Different request cannot reuse a probe directory')
 else:
  if any(p.name!='request.tmp' and not p.name.startswith('preempted-bootstrap-') for p in root.iterdir()):raise ValueError('Unknown existing output; preserve it and use a fresh run ID')
  if (root/'request.tmp').exists():(root/'request.tmp').rename(root/('preempted-bootstrap-'+uuid.uuid4().hex+'.tmp'))
  write_json(identity,request)
 result=root/'result.json'
 if not result.exists():return None
 complete=json.loads(result.read_text())
 if complete.get('request')!=request:raise ValueError('Completed result identity changed')
 seal(root)
 return complete
