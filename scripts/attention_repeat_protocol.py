"""Keep replication training seeds separate from the original data split."""
import hashlib,json
from pathlib import Path

def validate_configs(root,receipt,base):
 allowed={'seed','split_seed','attention_mode','run_name','purpose'}
 found=set();configs=[]
 for row in receipt['runs']:
  path=Path(root)/row['config'];raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Repeat config checksum changed')
  cfg=json.loads(raw)
  if cfg['split_seed']!=20261014 or cfg['seed'] not in [20261015,20261016]:raise ValueError('Training and split seeds must remain separate')
  if cfg['selection_metric']!='accuracy':raise ValueError('Require accuracy selection')
  if any(base.get(k)!=cfg.get(k) for k in (set(base)|set(cfg))-allowed):raise ValueError('Undeclared recipe change')
  pair=(cfg['seed'],cfg['attention_mode'])
  if pair in found:raise ValueError('Duplicate repeat condition')
  found.add(pair);configs.append(cfg)
 expected={(s,m) for s in [20261015,20261016] for m in ['native_causal','bidirectional_full_attention']}
 if found!=expected:raise ValueError('Incomplete matched seed pairs')
 return configs

def movement_probe(model,park_device='cpu',restore_device='cuda'):
 """Require exact output parity across actual CPU/GPU weight movement."""
 import torch
 text=model.model
 if hasattr(text,'language_model'):text=text.language_model
 def output():
  ids=torch.tensor([[1,2,3,4]],device=next(model.parameters()).device)
  with torch.no_grad():return text(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state.detach().cpu().clone()
 if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Backbone must be frozen')
 before=output();model.to(park_device);model.to(restore_device);after=output()
 if not torch.equal(before,after):raise ValueError('Offload/restore changed backbone output')
 return {'status':'passed','exact':True,'tokens':4,'park_device':park_device,'restore_device':restore_device,
         'hidden_sha256':hashlib.sha256(after.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
