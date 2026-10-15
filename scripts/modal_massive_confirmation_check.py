"""CPU-only paired recipe/split/schedule and actual initial-head fingerprint checks."""
import json,sys,subprocess,tempfile
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_source_addbacks import ROOT,ENV,image as base_image
app=modal.App('ac-worktrial-massive-confirmation-cpu');image=base_image
if modal.is_local():image=image.add_local_file(ROOT/'outputs/autoresearch/massive-confirmation-preparation.json','/project/massive-confirmation-plan.json')
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def check():
 import torch
 from safetensors.torch import save_file
 from train_joint import validate_config,epoch_windows,JointSchemaHead
 from training_data import load_splits,digest
 from prepare_massive_confirmation import prepare
 plan=json.loads(Path('/project/massive-confirmation-plan.json').read_text());counts={};heads={};heldouts=[]
 for entry in plan['runs']:
  path=Path('/project')/entry['config'];cfg=json.loads(path.read_text());validate_config(cfg)
  if digest(path)!=entry['config_sha256']:raise ValueError('Prepared config changed')
  manifest=Path('/project/source-addbacks')/entry['condition']/'manifest.json';splits,m=load_splits(manifest)
  if digest(manifest)!=entry['manifest_sha256'] or m['split_seed']!=cfg['split_seed'] or cfg['selection_metric']!='accuracy' or cfg['head_sdpa_backend']!='math':raise ValueError('Fixed protocol changed')
  heldouts.append({role:m['splits'][role]['points']['sha256'] for role in ['dev','calibration']})
  prepared=[(row,type('Encoded',(),{'questions':[None]})()) for row in splits['train']]
  windows=[epoch_windows(prepared,cfg['seed'],e,64) for e in range(5)]
  if len(splits['train'])!=6512 or sum(map(len,windows))!=510:raise ValueError('Decision/update count differs')
  counts[entry['run_id']]={'train':len(splits['train']),'dev':len(splits['dev']),'calibration':len(splits['calibration']),'updates':sum(map(len,windows))}
  with tempfile.TemporaryDirectory() as tmp:
   torch.manual_seed(cfg['seed']);head=JointSchemaHead(hidden_size=4096,**cfg['head']).float();p=Path(tmp)/'head.safetensors';save_file({k:v.detach().cpu().clone() for k,v in head.state_dict().items()},str(p));sha=digest(p)
   seed=str(cfg['seed'])
   if seed in heads and heads[seed]!=sha:raise ValueError('Paired initial heads differ')
   heads[seed]=sha
 if any(x!=heldouts[0] for x in heldouts):raise ValueError('Heldouts differ across source/control/seeds')
 cfg=json.loads((Path('/project')/plan['runs'][0]['config']).read_text())
 with tempfile.TemporaryDirectory() as tmp:
  torch.manual_seed(20261014);head=JointSchemaHead(hidden_size=4096,**cfg['head']).float();p=Path(tmp)/'head.safetensors';save_file({k:v.detach().cpu().clone() for k,v in head.state_dict().items()},str(p));original=digest(p)
 if original!='909b3c9ff1f565479d2bd81a1346564af2f5f0a433f2bf869fa9e7cad4ed573e':raise ValueError('CPU initialization does not reproduce actual original GPU-run initial tensor')
 if len(set(heads.values()))!=2:raise ValueError('New seeds must yield distinct initial heads')
 return {'status':'passed','counts':counts,'initial_head_sha256_by_training_seed':heads,'original_seed_initial_sha256_reproduced':original,'heldouts':heldouts[0],'source_sha256':{n:digest(Path('/project')/n) for n in ['scripts/prepare_massive_confirmation.py','scripts/modal_massive_confirmation_check.py']},'scope':'Actual CPU schema/loader/window and full Medium-head initialization checks; no GPU training or benchmark scoring.'}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/massive-confirmation-cpu.json'
 if out.exists():raise FileExistsError('Preserve prior proof')
 value=check.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
