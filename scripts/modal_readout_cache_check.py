"""Read-only CPU check of persistent27B shards and reusable nonbenchmark logits."""
import json,sys,hashlib,struct
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';app=modal.App('ac-worktrial-readout-cache-cpu')
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV);results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=2,memory=2048,timeout=600,retries=0,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results.read_only()})
def check():
 revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0';snapshot=Path('/models/hub/models--Qwen--Qwen3.8-27B/snapshots')/revision
 index_path=snapshot/'model.safetensors.index.json';index=json.loads(index_path.read_text());shards=sorted(set(index['weight_map'].values()));receipts=[];seen={};tensor_bytes=0
 for name in shards:
  path=snapshot/name;resolved=path.resolve(strict=True)
  anchor=Path('/models/hub/models--Qwen--Qwen3.8-27B/blobs').resolve(strict=True)
  if not resolved.is_relative_to(anchor):raise ValueError('Shard points outside resolved pinned model cache: '+str(resolved)+' versus '+str(anchor))
  size=resolved.stat().st_size
  with path.open('rb') as stream:
   raw=stream.read(8)
   if len(raw)!=8:raise ValueError('Truncated shard header')
   length=struct.unpack('<Q',raw)[0]
   if length<2 or length>20_000_000 or 8+length>=size:raise ValueError('Invalid shard header size')
   header=stream.read(length)
  metadata=json.loads(header);count=0
  for key,value in metadata.items():
   if key=='__metadata__':continue
   start,end=value['data_offsets']
   if key in seen or not 0<=start<end<=size-8-length:raise ValueError('Invalid/duplicate tensor offsets')
   if index['weight_map'].get(key)!=name:raise ValueError('Tensor index/shard mismatch')
   seen[key]=name;tensor_bytes+=end-start;count+=1
  receipts.append({'name':name,'blob':resolved.name,'file_bytes':size,'header_sha256':hashlib.sha256(header).hexdigest(),'tensors':count})
 if seen!=index['weight_map'] or tensor_bytes!=index['metadata']['total_size']:raise ValueError('Index coverage or tensor bytes differ')
 baseline=Path('/results/readout-fast-qwen38-27b-v1');checks=json.loads((baseline/'checksums.json').read_text());verified={}
 for name in ['config.json','completion.json','calibration-logits.jsonl','dev-logits.jsonl']:
  actual=hashlib.sha256((baseline/name).read_bytes()).hexdigest()
  if actual!=checks[name]:raise ValueError('Reusable baseline artifact changed: '+name)
  verified[name]=actual
 cfg=json.loads((baseline/'config.json').read_text());completion=json.loads((baseline/'completion.json').read_text())
 if cfg['backbone_revision']!=revision or completion['status']!='complete' or not completion['parameters_unchanged']:raise ValueError('Invalid baseline identity')
 return {'status':'passed','revision':revision,'resolved_snapshot':str(snapshot.resolve()),'resolved_blob_anchor':str(anchor),'shards':receipts,'shard_count':len(shards),'tensor_count':len(seen),'tensor_bytes':tensor_bytes,'index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),'baseline_artifacts_verified':verified,'GPU_allocated':False,'scope':'Symlink targets, complete safetensors headers/index/offset sizes and baseline cal/dev artifact SHA256. Does not rehash all tensor payload bytes or load the model; GPU smoke remains necessary.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/readout-persistent-cache-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior cache receipt')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='shards'},indent=2))
