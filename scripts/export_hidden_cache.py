"""Export every completed cache shard locally, then validate the final manifest."""
import argparse,concurrent.futures,hashlib,json,pathlib,time
import modal
from modal.config import _profile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 assert _profile=='ac-worktrial','Use MODAL_PROFILE=ac-worktrial'
 a=argparse.ArgumentParser();a.add_argument('--name',default='nulun3tv');a.add_argument('--remote',default='/hidden-cache/nulun3tv-qwen35-9b-3b3d8db82da7');args=a.parse_args()
 volume=modal.Volume.from_name('ac-worktrial-training-data',environment_name='worktrial-lance').hydrate()
 assert args.name.replace('-','').isalnum(),'Invalid export name'
 root=ROOT/'data/hidden-cache'/args.name;root.mkdir(parents=True,exist_ok=True);last=-1
 def read_json(name):return json.loads(b''.join(volume.read_file(args.remote+'/'+name)))
 def get(f):
  relative=pathlib.Path(f['path']);assert not relative.is_absolute() and '..' not in relative.parts
  p=root/relative
  if p.exists() and sha(p)==f['sha256']:return
  p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.download')
  with temp.open('wb') as out:
   for b in volume.read_file(args.remote+'/'+f['path']):out.write(b)
  if sha(temp)!=f['sha256']:raise ValueError('Export checksum mismatch '+f['path'])
  temp.replace(p)
 while True:
  try:
   progress=read_json('progress.json')
  except FileNotFoundError:time.sleep(20);continue
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(get,progress['shards']))
  count=len(progress['index'])
  if count!=last:print(json.dumps({'exported_records':count,'shards':len(progress['shards'])}),flush=True);last=count
  try:manifest=read_json('manifest.json')
  except FileNotFoundError:manifest=None
  if manifest and manifest.get('cached_head_backward_verified'):
   with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(get,manifest['files']))
   (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
   assert manifest['identity']['record_count']==5000
   receipt={'status':'complete','volume':'ac-worktrial-training-data','volume_id':volume.object_id,'url':'https://modal.com/id/'+volume.object_id,'environment':'worktrial-lance','profile':'ac-worktrial','remote_directory':args.remote,'local_directory':str(root),'files_verified':len(manifest['files']),'bytes':manifest['total_bytes'],'records':5000,'local_manifest_sha256':sha(root/'manifest.json'),'remote_manifest_sha256':hashlib.sha256(b''.join(volume.read_file(args.remote+'/manifest.json'))).hexdigest(),'all_file_hashes_verified':True,'identity':manifest['identity'],'parity_probes':manifest['parity_probes'],'cached_head_backward_verified':manifest['cached_head_backward_verified']}
   out=ROOT/('outputs/hidden-cache' if args.name=='nulun3tv' else 'outputs/hidden-cache-'+args.name)/'completion.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'complete','files':len(manifest['files']),'records':5000,'bytes':manifest['total_bytes']}),flush=True);break
  time.sleep(20)
if __name__=='__main__':main()
