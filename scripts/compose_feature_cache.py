"""Compose verified frozen caches and exact-request replay aliases on CPU."""
import collections,copy,json,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from training_data import canonical,digest,mounted_path

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def compose(parts,destination,rows,cfg,aliases=None,allow_source_subset=False):
 aliases=aliases or {};out=Path(destination);out.mkdir(parents=True,exist_ok=False)
 files=[];index={};lexical={};manifests=[];probes=[]
 common=['backbone_id','backbone_revision','schema_sha256','encoder_sha256','adapter_sha256','dtype','hidden_layer','extraction_batch_size','attention_implementation','seed','torch','transformers','hidden_size','no_truncation']
 def register(path):files.append({'path':str(path.relative_to(out)),'sha256':digest(path),'bytes':path.stat().st_size})
 for number,part in enumerate(parts):
  root=Path(part['directory']);mp=root/'manifest.json'
  if digest(mp)!=part['manifest_sha256']:raise ValueError('Parent manifest changed')
  manifest=json.loads(mp.read_text());identity=manifest['identity']
  if manifest['status']!='complete' or not manifest.get('cached_head_backward_verified'):raise ValueError('Parent cache incomplete')
  if manifests:
   for key in common:
    if identity[key]!=manifests[0]['identity'][key]:raise ValueError('Incompatible cache identity: '+key)
   if identity.get('attention_mode','native_causal')!=manifests[0]['identity'].get('attention_mode','native_causal'):raise ValueError('Attention modes differ')
  if identity['backbone_id']!=cfg['backbone_id'] or identity['backbone_revision']!=cfg['backbone_revision']:raise ValueError('Wrong backbone')
  if identity.get('attention_mode','native_causal')!=cfg.get('attention_mode','native_causal'):raise ValueError('Wrong attention mode')
  names=[entry['path'] for entry in manifest['files']]
  if len(names)!=len(set(names)) or not {'index.json','output-embedding-rows.safetensors'}<=set(names):raise ValueError('Incomplete/duplicate parent file inventory')
  for entry in manifest['files']:
   src=mounted_path(root,entry['path'])
   if digest(src)!=entry['sha256']:raise ValueError('Parent file checksum mismatch: '+entry['path'])
   if entry['path'] in ['index.json','output-embedding-rows.safetensors']:continue
   dest=out/f'parts/{number}'/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
   if digest(dest)!=entry['sha256']:raise ValueError('Copied file checksum mismatch')
   register(dest)
  dest=out/f'parts/{number}/manifest.json';write(dest,manifest);register(dest)
  for row in json.loads((root/'index.json').read_text()):
   if row['id'] in index:raise ValueError('Duplicate parent cache ID')
   if row['shard'] not in names:raise ValueError('Unverified parent tensor shard')
   row=copy.deepcopy(row);row['shard']=f"parts/{number}/"+row['shard'];index[row['id']]=row
  lex=load_file(str(root/'output-embedding-rows.safetensors'))
  for token,value in zip(lex['token_ids'].tolist(),lex['rows']):
   if token in lexical and not torch.equal(lexical[token],value):raise ValueError('Shared output embedding row differs')
   lexical[token]=value.clone()
  manifests.append(manifest);probes.extend(manifest['parity_probes'])
 for alias,parent in aliases.items():
  if alias in index or parent not in index or parent in aliases:raise ValueError('Invalid replay alias')
  index[alias]=dict(copy.deepcopy(index[parent]),id=alias,replay_parent_id=parent)
 target_ids={r['id'] for r in rows}
 if allow_source_subset:
  if not target_ids<=set(index):raise ValueError('Requested subset ID missing from source cache')
  index={rid:index[rid] for rid in target_ids}
 if len({r['id'] for r in rows})!=len(rows) or set(index)!={r['id'] for r in rows}:raise ValueError('Dataset/cache ID set differs')
 ordered=[]
 for row in rows:
  entry=index[row['id']]
  import hashlib
  request_hash=hashlib.sha256(canonical(row['request']).encode()).hexdigest()
  if request_hash!=entry['request_sha256'] or row['targets']!=entry['targets']:raise ValueError('Replay request or targets differ')
  if row['group_id']!=entry['group_id'] or row['source']!=entry['source']:raise ValueError('Replay source/group differs')
  ordered.append(entry)
 ids=sorted(lexical);lexpath=out/'output-embedding-rows.safetensors'
 save_file({'token_ids':torch.tensor(ids,dtype=torch.long),'rows':torch.stack([lexical[i] for i in ids])},str(lexpath));register(lexpath)
 write(out/'index.json',ordered);register(out/'index.json')
 identity=dict(manifests[0]['identity'],dataset_sha256=cfg['dataset']['sha256'],record_count=len(rows),decision_count=sum(len(r['targets']) for r in rows),total_tokens=sum(r['token_count'] for r in ordered),training_source_config=cfg,composition='Unmodified source tensors; preserved extraction batches; exact-request replay aliases share tensor offsets',parent_manifest_sha256=[p['manifest_sha256'] for p in parts],replay_aliases=aliases)
 identity['feature_bytes']=identity['total_tokens']*identity['hidden_size']*2
 write(out/'extraction-config.json',identity);register(out/'extraction-config.json')
 manifest={'status':'complete','identity':identity,'files':files,'parity_probes':probes,'cached_head_backward_verified':True,'total_bytes':sum(f['bytes'] for f in files),'source_counts':dict(collections.Counter(r['source'] for r in ordered)),'note':'feature_bytes counts logical examples; replay aliases share physical tensor storage. Parent files copied byte-for-byte, with shared lexical rows equality checked.'}
 write(out/'manifest.json',manifest);return manifest
