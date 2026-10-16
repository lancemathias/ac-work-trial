"""Reuse one frozen backbone per worker, with explicit attention reset."""
import contextlib,contextvars,gc,json,time
_active=contextvars.ContextVar('resident_backbone',default=None)

def reset_attention(model):
 text=model.model
 if hasattr(text,'language_model'):text=text.language_model
 handle=getattr(text,'_research_attention_hook',None)
 if handle is not None:handle.remove();delattr(text,'_research_attention_hook')
 for layer in getattr(text,'layers',[]):
  if getattr(layer,'block_type',None)=='full_attention':layer.self_attn.is_causal=True

class BackbonePool:
 def __init__(self,processor_loader=None,model_loader=None):
  self.processor_loader=processor_loader;self.model_loader=model_loader
  self.key=None;self.model=None;self.processor=None;self.processor_key=None;self.loads=0;self.hits=0
 def get_processor(self,cfg):
  key=(cfg['backbone_id'],cfg['backbone_revision'],cfg.get('model_cache','/models/hub'))
  if self.processor_key!=key:
   if self.processor_loader is None:
    from transformers import AutoProcessor
    loader=AutoProcessor.from_pretrained
   else:loader=self.processor_loader
   self.processor=loader(key[0],revision=key[1],cache_dir=key[2],local_files_only=True);self.processor_key=key
  return self.processor
 def get_model(self,cfg):
  import torch
  key=(cfg['backbone_id'],cfg['backbone_revision'],cfg.get('model_cache','/models/hub'))
  began=time.monotonic()
  if self.model is None or self.key!=key:
   self.close()
   if self.model_loader is None:
    from transformers import Qwen3_5ForConditionalGeneration
    loader=Qwen3_5ForConditionalGeneration.from_pretrained
   else:loader=self.model_loader
   self.model=loader(key[0],revision=key[1],cache_dir=key[2],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'}).requires_grad_(False).eval()
   self.key=key;self.loads+=1;event='resident_backbone_load'
  else:self.hits+=1;event='resident_backbone_reuse'
  reset_attention(self.model)
  if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):raise ValueError('Resident backbone must stay frozen')
  print(json.dumps({'event':event,'backbone_id':key[0],'loads':self.loads,'hits':self.hits,'seconds':time.monotonic()-began}),flush=True)
  return self.model
 def close(self):
  import torch
  if self.model is not None:reset_attention(self.model)
  self.model=None;self.key=None;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 @contextlib.contextmanager
 def activate(self):
  token=_active.set(self)
  try:yield self
  finally:_active.reset(token)

def get_processor(cfg):
 pool=_active.get()
 if pool is not None:return pool.get_processor(cfg)
 from transformers import AutoProcessor
 return AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir=cfg.get('model_cache','/models/hub'),local_files_only=True)

def get_model(cfg):
 pool=_active.get()
 return (pool if pool is not None else BackbonePool()).get_model(cfg)
