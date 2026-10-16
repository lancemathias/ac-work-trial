"""Recover the pinned original GitHub training inputs from tracked receipts."""
import hashlib
import json
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sources=json.loads((ROOT/'outputs/data-releases/v4-expanded-noul/source-inputs.json').read_text())['sources']
for source in sources:
    if 'files' not in source:continue
    for entry in source['files']:
        target=ROOT/entry['path']
        if not target.exists():
            if not entry['url'].startswith('https://raw.githubusercontent.com/'+source['repo']+'/'+source['revision']+'/'):raise ValueError('Expected immutable official GitHub URL')
            payload=urllib.request.urlopen(entry['url'],timeout=180).read()
            if hashlib.sha256(payload).hexdigest()!=entry['sha256']:raise ValueError('Downloaded source checksum mismatch')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Pinned local source checksum mismatch')
    folder=ROOT/Path(source['files'][0]['path']).parent
    (folder/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    print(source['repo'],source['revision'],'verified')
