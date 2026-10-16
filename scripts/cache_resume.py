"""Fail-closed cache resume identity checks with separate physical provenance."""
import copy,hashlib,json

def compatible_identity(previous,current,previous_backend,current_backend):
 def digest(value):return hashlib.sha256(value).hexdigest()
 if digest(previous_backend)!=previous['backbone_metadata']['backend_settings_sha256']:raise ValueError('Original backend metadata checksum differs')
 if digest(current_backend)!=current['backbone_metadata']['backend_settings_sha256']:raise ValueError('Current backend metadata checksum differs')
 def semantics(raw):
  backend=json.loads(raw);fields=backend['nvidia_smi'].split(', ')
  if len(fields)!=4 or not fields[1].startswith('GPU-'):raise ValueError('Unexpected GPU provenance format')
  backend['nvidia_smi']=[fields[0],fields[2],fields[3]]
  return backend
 if semantics(previous_backend)!=semantics(current_backend):raise ValueError('Computational backend differs beyond physical GPU UUID')
 before,after=copy.deepcopy(previous),copy.deepcopy(current)
 for value in [before,after]:value['backbone_metadata'].pop('backend_settings_sha256')
 if 'attention_mode' not in before and 'attention_hook_sha256' not in before and 'attention_mode' in after:
  # Legacy native extractor had no attention hook; current native mode is a no-op.
  if (after.get('attention_mode')!='native_causal' or not after.get('attention_hook_sha256')
      or before.get('training_source_config',{}).get('attention_mode','native_causal')!='native_causal'):
   raise ValueError('Legacy cache only supports native attention identity migration')
  after.pop('attention_mode');after.pop('attention_hook_sha256')
 if before!=after:
  keys=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
  raise ValueError('Existing cache computation identity differs: '+','.join(keys))
 return copy.deepcopy(previous)

def require_exact_replay(actual,cached):
 import torch
 if actual.shape!=cached.shape or actual.dtype!=cached.dtype or not torch.equal(actual.cpu(),cached.cpu()):raise ValueError('Replacement worker cached-feature parity failed')

def require_extraction_phase(previous,manual_marker=None):
 if previous['phase']=='extracting_frozen_features':return
 # Explicitly stopped prior manual extraction receives Modal RemoteError.
 if (previous['phase']=='failed' and previous.get('manual_recovery') is True
     and previous.get('error')=='RemoteError' and previous.get('message')==''
     and manual_marker and manual_marker['previous']['phase']=='extracting_frozen_features'
     and manual_marker['previous']['run_id']==previous['run_id']
     and manual_marker['recovery_commit']==previous['commit']):return
 raise ValueError('Recovery only allowed for diagnosed extraction interruption')
