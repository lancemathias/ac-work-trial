"""Preserve failed cache files locally with readback hashes, then delete only that remote directory."""
import concurrent.futures,hashlib,json,pathlib
import modal
from modal.config import _profile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert _profile=='ac-worktrial'
 remote='/hidden-cache/qwen35-9b-additions-2500-batch4-cf5d3798fefb'
 local=ROOT/'data/hidden-cache/qwen35-9b-additions-2500-batch4';local.mkdir(parents=True,exist_ok=True)
 volume=modal.Volume.from_name('ac-worktrial-training-data',environment_name='worktrial-lance').hydrate()
 entries=volume.listdir(remote,recursive=True)
 print('entries',len(entries),flush=True)
 specs=[]
 for e in entries:
  if str(e.type).lower().endswith('directory') or getattr(e.type,'name','')=='DIRECTORY':continue
  path=e.path;relative=path.removeprefix(remote.lstrip('/')+'/').removeprefix(remote+'/')
  if relative==path:raise ValueError(('Unexpected path',path))
  specs.append({'path':path,'relative':relative,'size':e.size})
 try:progress=json.loads(b''.join(volume.read_file(remote+'/progress.json')))
 except FileNotFoundError:raise ValueError('Failed cache progress missing; inspect before cleanup')
 shards={s['path']:s['sha256'] for s in progress['shards']}
 def get(s):
  p=local/s['relative'];p.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();tmp=p.with_suffix(p.suffix+'.archive-download')
  with tmp.open('wb') as f:
   for b in volume.read_file(s['path']):h.update(b);f.write(b)
  if s['relative'] in shards:assert h.hexdigest()==shards[s['relative']]
  tmp.replace(p);assert sha(p)==h.hexdigest();return dict(s,sha256=h.hexdigest())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:verified=list(pool.map(get,specs))
 receipt={'failed_app':'ap-dReKH7p4rtlOAgWllCFgQe','replacement_app':'ap-AohSa1P1m6kwN8x9dvZIvR','profile':'ac-worktrial','environment':'worktrial-lance','volume':'ac-worktrial-training-data','remote_directory':remote,'local_directory':str(local),'preserved_records':len(progress['index']),'files':verified,'all_local_hashes_verified':True,'remote_removed':False}
 out=ROOT/'outputs/hidden-cache-qwen35-9b-additions-2500-batch4/cleanup.json';out.write_text(json.dumps(receipt,indent=2)+'\n')
 volume.remove_file(remote,recursive=True);receipt['remote_removed']=True;out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
