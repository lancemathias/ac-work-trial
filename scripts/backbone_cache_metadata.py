"""Capture immutable inputs, actual backend settings, and observed attention masks."""
import collections,hashlib,importlib.metadata,inspect,json,os,pathlib,platform,shutil,subprocess,sys
import torch

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def tensor_info(t):
 if t is None:return {'present':False}
 raw=t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 return {'present':True,'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(raw).hexdigest(),'numel':t.numel()}
def collect_metadata(backbone,processor,cfg,out):
 from huggingface_hub import hf_hub_download
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 out=pathlib.Path(out);directory=out/'backbone-metadata';directory.mkdir(exist_ok=True)
 snapshot=pathlib.Path(hf_hub_download(repo_id=cfg['backbone_id'],filename='config.json',revision=cfg['backbone_revision'],cache_dir=cfg['model_cache'],local_files_only=True)).parent
 for filename in ['.gitattributes','LICENSE','README.md']:
  hf_hub_download(repo_id=cfg['backbone_id'],filename=filename,revision=cfg['backbone_revision'],cache_dir=cfg['model_cache'])
 files=[]
 for p in sorted(snapshot.rglob('*')):
  if not p.is_file():continue
  entry={'path':str(p.relative_to(snapshot)),'bytes':p.stat().st_size,'sha256':sha(p)};files.append(entry)
  if p.suffix not in {'.safetensors','.bin','.pt','.pth'}:
   dest=directory/'hf-assets'/p.relative_to(snapshot);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 sourcefiles=[]
 for name,module in sorted(sys.modules.copy().items()):
  path=getattr(module,'__file__',None)
  if not name.startswith('transformers.') or not path or not path.endswith('.py'):continue
  p=pathlib.Path(path)
  if not p.is_file():continue
  dest=directory/'implementation'/pathlib.Path(*name.split('.')).with_suffix('.py');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  sourcefiles.append({'module':name,'path':str(dest.relative_to(out)),'sha256':sha(p)})
 parameters=[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad} for n,p in backbone.named_parameters()]
 tokenizer=processor.tokenizer
 tok={'class':type(tokenizer).__name__,'special_tokens_map':tokenizer.special_tokens_map,'all_special_tokens':tokenizer.all_special_tokens,'all_special_ids':tokenizer.all_special_ids,'pad_token_id':tokenizer.pad_token_id,'padding_side':tokenizer.padding_side,'truncation_side':tokenizer.truncation_side,'model_max_length':tokenizer.model_max_length,'init_kwargs':tokenizer.init_kwargs,'vocab_sha256':hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()}
 backend={'cuda':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'flash_sdp_enabled':torch.backends.cuda.flash_sdp_enabled(),'math_sdp_enabled':torch.backends.cuda.math_sdp_enabled(),'mem_efficient_sdp_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),'cudnn_sdp_enabled':torch.backends.cuda.cudnn_sdp_enabled(),'environment':{k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED','CUDA_VISIBLE_DEVICES','NVIDIA_TF32_OVERRIDE']},'gpu':torch.cuda.get_device_name(),'compute_capability':list(torch.cuda.get_device_capability()),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
 kernels={}
 text=backbone.model.language_model if hasattr(backbone.model,'language_model') else backbone.model
 for n,m in text.named_modules():
  if 'GatedDelta' not in type(m).__name__:continue
  kernels[n]={'class':type(m).__name__,'functions':{k:f'{getattr(getattr(m,k),"__module__",None)}.{getattr(getattr(m,k),"__name__",None)}' for k in ['causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'] if hasattr(m,k)}}
 values={'model-config.json':backbone.config.to_dict(),'tokenizer-metadata.json':tok,'parameter-inventory.json':parameters,'hf-snapshot-files.json':files,'implementation-files.json':sourcefiles,'backend-settings.json':backend,'linear-attention-kernels.json':kernels,'package-versions.json':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'process.json':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'model_class':type(backbone).__name__,'text_model_class':type(text).__name__,'backbone_training_mode':backbone.training,'text_model_training_mode':text.training,'forward_use_cache':False,'generation_sampling':False,'position_ids_argument':'omitted: exact Qwen text-model default arange starting at zero, no past cache; implementation preserved','input_attention_mask':'int64 [1,T], all ones, no padding','attention_semantics':'Observe mask factory outputs and real SDPA is_causal flags per record. Linear attention uses the preserved gated-delta recurrent implementation, not bidirectional full attention.'}}
 for name,value in values.items():(directory/name).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
 core=[]
 for name in ['_C','lib/libtorch_cuda.so','lib/libtorch_cpu.so']:
  if name=='_C':p=pathlib.Path(torch._C.__file__)
  else:p=pathlib.Path(torch.__file__).parent/name
  if p.exists():core.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 (directory/'torch-binary-checksums.json').write_text(json.dumps(core,indent=2)+'\n')
 return {'metadata_directory':'backbone-metadata','hf_snapshot_files':len(files),'hf_weight_files_hashed':sum(f['path'].endswith('.safetensors') for f in files),'implementation_modules_preserved':len(sourcefiles),'layer_types':getattr(text.config,'layer_types',None),'model_config_sha256':sha(directory/'model-config.json'),'backend_settings_sha256':sha(directory/'backend-settings.json')}

class AttentionRecorder:
 """Observe masks/flags without changing arguments or model numerical operations."""
 def __init__(self):
  from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
  self.qwen=qwen;self.enabled=False;self.events=[];self.originals={}
 def install(self):
  for name in ['create_causal_mask','create_recurrent_attention_mask']:
   original=getattr(self.qwen,name);self.originals[name]=original
   def wrap(*args,_name=name,_original=original,**kwargs):
    result=_original(*args,**kwargs)
    if self.enabled:self.events.append({'operation':_name,'output_mask':tensor_info(result)})
    return result
   setattr(self.qwen,name,wrap)
  original=torch.nn.functional.scaled_dot_product_attention;self.originals['sdpa']=original
  def sdpa(*args,**kwargs):
   if self.enabled:
    mask=kwargs.get('attn_mask',args[3] if len(args)>3 else None)
    event={'operation':'scaled_dot_product_attention','query_shape':list(args[0].shape),'key_shape':list(args[1].shape),'value_shape':list(args[2].shape),'explicit_mask':tensor_info(mask),'is_causal':bool(kwargs.get('is_causal',args[5] if len(args)>5 else False)),'dropout_p':float(kwargs.get('dropout_p',args[4] if len(args)>4 else 0.0)),'scale':kwargs.get('scale'),'enable_gqa':bool(kwargs.get('enable_gqa',False))}
    self.events.append(event)
   return original(*args,**kwargs)
  torch.nn.functional.scaled_dot_product_attention=sdpa
 def begin(self):self.events=[];self.enabled=True
 def end(self):
  self.enabled=False;counts=collections.Counter(json.dumps(e,sort_keys=True) for e in self.events)
  return [dict(json.loads(e),calls=n) for e,n in counts.items()]
 def restore(self):
  for name in ['create_causal_mask','create_recurrent_attention_mask']:setattr(self.qwen,name,self.originals[name])
  torch.nn.functional.scaled_dot_product_attention=self.originals['sdpa']
