"""Explicit full-attention ablation; keep hybrid recurrent paths and padding."""
import torch
NATIVE='native_causal'
BIDIRECTIONAL='bidirectional_full_attention'
def install_attention_mode(backbone,mode=NATIVE):
 if mode==NATIVE:return None
 if mode!=BIDIRECTIONAL:raise ValueError('Unknown attention mode: '+mode)
 text=backbone.model
 if hasattr(text,'language_model'):text=text.language_model
 if getattr(text,'_research_attention_hook',None) is not None:raise ValueError('Attention hook already installed')
 from transformers.models.qwen3_5.modeling_qwen3_5 import create_recurrent_attention_mask
 def replace_mask(module,args,kwargs):
  if args:raise ValueError('Ablation expects keyword-only text inputs')
  if kwargs.get('past_key_values') is not None or kwargs.get('use_cache',False):raise ValueError('Ablation supports full prefill only')
  ids=kwargs.get('input_ids');embeds=kwargs.get('inputs_embeds')
  if embeds is None:
   if ids is None:raise ValueError('Missing text inputs')
   embeds=module.embed_tokens(ids)
  batch,length=embeds.shape[:2];padding=kwargs.get('attention_mask')
  if padding is None:padding=torch.ones((batch,length),dtype=torch.long,device=embeds.device)
  if not isinstance(padding,torch.Tensor) or padding.shape!=(batch,length):raise ValueError('Expected 2D padding mask')
  if not torch.all((padding==0)|(padding==1)):raise ValueError('Padding mask must be binary')
  # True means an allowed key for PyTorch SDPA. Keep padded keys inaccessible.
  full=padding[:,None,None,:].bool().expand(batch,1,length,length)
  recurrent=create_recurrent_attention_mask(config=module.config,inputs_embeds=embeds,attention_mask=padding,past_key_values=None,position_ids=None)
  kwargs=dict(kwargs,attention_mask={'full_attention':full,'linear_attention':recurrent})
  return args,kwargs
 for layer in text.layers:
  if layer.block_type=='full_attention':layer.self_attn.is_causal=False
 handle=text.register_forward_pre_hook(replace_mask,with_kwargs=True)
 text._research_attention_hook=handle
 return handle
