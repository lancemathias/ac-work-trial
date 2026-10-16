"""Wait for the launched v4 cache, download its tensors, and verify every file."""
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path
os.environ['MODAL_PROFILE']='ac-worktrial'
os.environ['MODAL_ENVIRONMENT']='worktrial-lance'
import modal
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'outputs/data-releases/v4-expanded-noul'
import time
while not (FOLDER/'cache-launch.json').exists():
    print('Waiting for queued v4 extension to launch',flush=True)
    time.sleep(45)
launch=json.loads((FOLDER/'cache-launch.json').read_text())
call=modal.FunctionCall.from_id(launch['function_call_id'])
try:
    while True:
        try:
            result=call.get(timeout=45)
            break
        except TimeoutError:
            print('Waiting for v4 27B cache completion',flush=True)
    (FOLDER/'cache-completion.json').write_text(json.dumps(result,indent=2)+'\n')
    exports=ROOT/'outputs/model-artifacts';exports.mkdir(exist_ok=True)
    archive=exports/Path(result['archive_path']).name
    if not archive.exists():
        subprocess.run([str(ROOT/'.venv/bin/modal'),'volume','get','ac-worktrial-training-results',result['archive_path'].removeprefix('/results/'),str(archive),'--profile','ac-worktrial','--env','worktrial-lance'],check=True)
    def hashed(stream):
        sha=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
        return sha.hexdigest()
    with archive.open('rb') as stream:actual=hashed(stream)
    if actual!=result['sha256'] or archive.stat().st_size!=result['bytes']:raise ValueError('Local tensor archive checksum/size mismatch')
    verified=0
    with tarfile.open(archive,'r:gz') as tar:
        prefix=result['result']['version']  # Manifest is v4; archive directory has its full release name.
        top=next(m.name.split('/')[0] for m in tar.getmembers() if m.isfile())
        checksums=json.load(tar.extractfile(top+'/checksums.json'))
        for name,expected in checksums.items():
            stream=tar.extractfile(top+'/'+name)
            if stream is None or hashed(stream)!=expected:raise ValueError('Archived feature file mismatch: '+name)
            verified+=1
        manifest=json.load(tar.extractfile(top+'/manifest.json'))
        if manifest['train_views']!=21524 or manifest['new_views']!=4769 or manifest['backbone_id']!='Qwen/Qwen3.8-27B':raise ValueError('Unexpected v4 cache inventory')
    receipt={'status':'complete','archive':str(archive.relative_to(ROOT)),'sha256':actual,'bytes':archive.stat().st_size,'verified_archived_files':verified,'train_views':21524,'new_27b_views':4769,'reused_previous_v4_views':16755,'dev_calibration_unchanged':True,'backbone_id':manifest['backbone_id'],'backbone_revision':manifest['backbone_revision'],'no_head_training':True}
    (FOLDER/'local-cache-export.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[str((FOLDER/name).relative_to(ROOT)) for name in ['cache-completion.json','local-cache-export.json']]
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','--only',*paths,'-m','Preserve and verify expanded v4 Noul feature cache'],cwd=ROOT,check=True)
    print(json.dumps(receipt,indent=2),flush=True)
except Exception as exc:
    (FOLDER/'finalization-failure.json').write_text(json.dumps({'error':type(exc).__name__,'message':str(exc)},indent=2)+'\n')
    raise
