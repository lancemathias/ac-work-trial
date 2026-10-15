"""Describe actual optional GPU kernels for reproducible cache identity."""
import inspect
import importlib.metadata as metadata

def actual_implementation(fn):
 while fn is not None:
  if inspect.isfunction(fn):
   values=inspect.getclosurevars(fn).nonlocals
   if 'implementation' in values:return values['implementation']
  wrapped=getattr(fn,'__wrapped__',None)
  if wrapped is None and inspect.isfunction(fn):return fn
  fn=wrapped
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
 if not paths['torch_chunk_gated_delta_rule'].startswith('fla.'):
  raise ValueError('Installed optimized kernels were not dispatched')
 return {'mode':'both' if paths['causal_conv1d_fn'].startswith('causal_conv1d.') else 'linear_only','versions':versions,'implementations':paths}


def configure_optimized_runtime(mode='both'):
 if mode not in ['both','linear_only']:raise ValueError('Unsupported optimized runtime')
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 if mode=='linear_only':
  for name in ['causal_conv1d_fn','causal_conv1d_update']:
   setattr(qwen,name,inspect.unwrap(getattr(qwen,name)))
 identity=optimized_runtime_identity()
 if identity is None or identity['mode']!=mode:raise ValueError('Optimized runtime not configured as requested')
 return identity
