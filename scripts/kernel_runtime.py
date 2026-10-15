"""Describe actual optional GPU kernels for reproducible cache identity."""
import inspect
import importlib.metadata as metadata

def actual_implementation(fn):
 while fn is not None:
  if inspect.isfunction(fn):
   values=inspect.getclosurevars(fn).nonlocals
   if 'implementation' in values:return values['implementation']
  fn=getattr(fn,'__wrapped__',None)
 raise ValueError('Cannot prove selected kernel implementation')


def optimized_runtime_identity():
 try:
  versions={n:metadata.version(n) for n in ['causal-conv1d','fla-core','flash-linear-attention','einops']}
 except metadata.PackageNotFoundError:
  return None
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 names=['causal_conv1d_fn','causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']
 implementations={n:actual_implementation(getattr(qwen,n)) for n in names}
 paths={n:f.__module__+'.'+f.__name__ for n,f in implementations.items()}
 if not paths['causal_conv1d_fn'].startswith('causal_conv1d.') or not paths['torch_chunk_gated_delta_rule'].startswith('fla.'):
  raise ValueError('Installed optimized kernels were not dispatched')
 return {'versions':versions,'implementations':paths}
