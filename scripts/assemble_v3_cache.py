"""Assemble checked batch-four v2 and additions caches without new model forwards."""
import collections,hashlib,json,pathlib,shutil,time
import torch
from safetensors.torch import load_file,save_file

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def run(volume):
 root=pathlib.Path('/data');dataset=root/'dataset-v3-ef15c46f4919/train-v3.jsonl';out=root/'hidden-cache/qwen35-9b-v3-batch4-ef15c46f4919'
 parents=[root/'hidden-cache/qwen35-9b-v2.1-batch4-dc951c750839',root/'hidden-cache/qwen35-9b-v3-additions-batch4-88993d2c382d']
 for _ in range(360):
  volume.reload()
  if all((p/'manifest.json').exists() and json.loads((p/'manifest.json').read_text()).get('cached_head_backward_verified') for p in parents):break
  time.sleep(20)
 else:raise TimeoutError('Parent extraction incomplete after two hours; no complete composite published')
 assert sha(dataset)=='ef15c46f49192cafe5d35a1b0567e1a33cc13807fb5c2ca4d50e1fdbc6beec81'
 rows=[json.loads(l) for l in dataset.open()];assert len(rows)==7928
 out.mkdir(parents=True,exist_ok=True);files=[];allindex={};embeddings={};manifests=[];shards=[]
 def register(p):files.append({'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size})
 for part,parent in enumerate(parents):
  m=json.loads((parent/'manifest.json').read_text());manifests.append(m)
  if part:
   for k in ['backbone_id','backbone_revision','schema_sha256','encoder_sha256','adapter_sha256','dtype','hidden_layer','extraction_batch_size','attention_implementation','seed','torch','transformers','hidden_size']:
    assert m['identity'][k]==manifests[0]['identity'][k],k
  for f in m['files']:
   src=parent/f['path'];assert sha(src)==f['sha256'],str(src)
   # Keep source manifests/metadata plus tensors; merged index and embedding rows are rebuilt.
   if f['path'] in ['index.json','output-embedding-rows.safetensors']:continue
   dest=out/f'parts/{part}'/f['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(dest)==f['sha256'];register(dest)
   if '/shards/' in '/'+f['path']:shards.append(dict(f,path=str(dest.relative_to(out))))
  write(out/f'parts/{part}/manifest.json',m);register(out/f'parts/{part}/manifest.json')
  for r in json.loads((parent/'index.json').read_text()):
   assert r['id'] not in allindex;r['shard']=f"parts/{part}/"+r['shard'];r['source_cache_part']=part;allindex[r['id']]=r
  lex=load_file(str(parent/'output-embedding-rows.safetensors'))
  for token,row in zip(lex['token_ids'].tolist(),lex['rows']):
   if token in embeddings:assert torch.equal(embeddings[token],row),token
   else:embeddings[token]=row.clone()
 assert set(allindex)=={r['id'] for r in rows}
 index=[allindex[r['id']] for r in rows]
 for row,entry in zip(rows,index):
  request={'model':'jev-latest','state':row['state'],'questions':{'q':row['question']}}
  canonical=json.dumps(request,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
  assert hashlib.sha256(canonical.encode()).hexdigest()==entry['request_sha256']
  gold='true' if row['label'] is True else 'false' if row['label'] is False else row['label'];assert entry['targets']=={'q':gold}
 ids=sorted(embeddings);save_file({'token_ids':torch.tensor(ids,dtype=torch.long),'rows':torch.stack([embeddings[i] for i in ids])},str(out/'output-embedding-rows.safetensors'));register(out/'output-embedding-rows.safetensors')
 write(out/'index.json',index);register(out/'index.json')
 identity=dict(manifests[0]['identity'],dataset_sha256=sha(dataset),record_count=7928,decision_count=7928,total_tokens=sum(r['token_count'] for r in index),dataset_version='v3',composition='Preserved original per-batch states; no new forward pass or rebatching during assembly',parent_cache_identities=[m['identity'] for m in manifests]);identity['feature_bytes']=identity['total_tokens']*identity['hidden_size']*2
 write(out/'extraction-config.json',identity);register(out/'extraction-config.json')
 manifest={'status':'complete','identity':identity,'files':files,'total_bytes':sum(f['bytes'] for f in files),'source_counts':dict(collections.Counter(r['source'] for r in index)),'parity_probes':[q for m in manifests for q in m['parity_probes']],'cached_head_backward_verified':all(m['cached_head_backward_verified'] for m in manifests),'validation':'All parent files checked; option embedding rows identical on overlap; all 7928 dataset IDs/request hashes/gold targets checked; metadata preserved by part.','note':'Composite cache preserves original batch membership and per-part hardware metadata. No claim of re-forward bitwise reproducibility.'}
 write(out/'progress.json',{'identity':identity,'index':index,'shards':shards,'probes':manifest['parity_probes']});write(out/'manifest.json',manifest);volume.commit()
 return {'status':'complete','records':7928,'bytes':manifest['total_bytes'],'remote_directory':str(out.relative_to(root))}
