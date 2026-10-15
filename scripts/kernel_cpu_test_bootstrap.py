"""Keep CPU fixtures on reference ops when CUDA-only optional kernels are installed."""
import inspect,os,sys,unittest
sys.path.insert(0,os.getcwd())
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
for name in ['causal_conv1d_fn','causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']:
 setattr(qwen,name,inspect.unwrap(getattr(qwen,name)))
unittest.main(module=None,argv=['cpu-kernel-fixtures','-v',*sys.argv[1:]])
