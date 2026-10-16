"""Select head attention without changing frozen-backbone attention."""
from contextlib import nullcontext

def configure_head(head,cfg):
 mode=cfg.get('head_sdpa_backend','auto')
 if mode not in {'auto','math'}:raise ValueError('Unsupported head SDPA backend')
 head._research_sdpa_backend=mode
 return head

def attention_context(head):
 mode=getattr(head,'_research_sdpa_backend','auto')
 if mode=='auto':return nullcontext()
 if mode!='math':raise ValueError('Unsupported head SDPA backend')
 from torch.nn.attention import sdpa_kernel,SDPBackend
 return sdpa_kernel([SDPBackend.MATH])
