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
 if before!=after:raise ValueError('Existing cache computation identity differs')
 return copy.deepcopy(previous)

def require_exact_replay(actual,cached):
 import torch
 if actual.shape!=cached.shape or actual.dtype!=cached.dtype or not torch.equal(actual.cpu(),cached.cpu()):raise ValueError('Replacement worker cached-feature parity failed')
