"""Inspect the pinned hybrid attention implementation on CPU before ablation."""
import json,inspect
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance'
image=modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
app=modal.App('ac-worktrial-attention-ablation-inspection')
@app.function(image=image,cpu=2,memory=4096,timeout=300)
def probe():
 from transformers.models.qwen3_5 import modeling_qwen3_5 as m
 values={}
 for name in ['Qwen3_5TextModel','Qwen3_5Attention','Qwen3_5DecoderLayer','Qwen3_5GatedDeltaNet']:
  values[name]=inspect.getsource(getattr(m,name))
 return values
@app.local_entrypoint()
def main():
 from modal.config import config,_profile
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=probe.remote();p=ROOT/'outputs/autoresearch/pinned-attention-source.json';p.write_text(json.dumps(result,indent=2)+'\n')
 for name,source in result.items():
  lines=source.splitlines()
  print(name)
  for i,line in enumerate(lines):
   if any(word in line for word in ['causal','attention_mask','layer_types','masking','linear_attention','full_attention']):print(i+1,line)
