"""Read-only CPU assessment of fully emitted thinking cal/dev; never reads test."""
import json,sys,subprocess,hashlib
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image as base_image,results,secret
image=base_image.add_local_file(ROOT/'outputs/autoresearch/experiment-cards.json','/project/experiment-cards.json') if modal.is_local() else base_image
app=modal.App('ac-worktrial-selective-thinking-development')
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2,volumes={'/results':results},secrets=[secret])
def assess(commit):
 from evaluate_cached_mixture import verified_rows,verified_json
 from assess_selective_thinking import assess as run
 from training_data import load_splits,digest,canonical
 from head_probe_io import prepare_run,write_json,seal
 from train_joint import fit_temperature,metrics,slice_metrics
 import wandb
 native=Path('/results/readout-fast-qwen38-27b-v1');reverse=Path('/results/readout-native-reverse-pool-v1');parent=Path('/results/readout-bounded-thinking128-staged-v2')
 # There must be one attempt with a completed dev gate; partial rows never qualify.
 ready=[p.parent for p in (parent/'attempts').glob('*/development.json')]
 if not ready:return {'status':'waiting_for_complete_thinking_development','test_read':False,'GPU_allocated':False}
 if len(ready)!=1:raise ValueError('Ambiguous thinking attempt; explicit audit required')
 thinking=ready[0];cfg=json.loads((thinking/'config.json').read_text());old=verified_json(native,'config.json')
 if any(cfg['settings'][k]!=old[k] for k in ['backbone_id','backbone_revision']):raise ValueError('Backbone differs')
 if cfg['settings']!=json.loads(Path('/project/configs/bounded-thinking128.json').read_text()):raise ValueError('Thinking budget or settings changed')
 if cfg['parent_checkpoint_sha256']!=old['checkpoint_sha256'] or cfg['panel_sha256']!=old['panel_sha256']:raise ValueError('Thinking parent identity differs')
 cards=json.loads(Path('/project/experiment-cards.json').read_text())['cards'];card=next(c for c in cards if c['id']=='pretrained-order-disagreement-thinking128')
 splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json');inputs={};hashes={}
 for role,count in [('calibration',857),('dev',809)]:
  a=verified_rows(native,role+'-logits.jsonl');b=verified_rows(reverse,'reverse-'+role+'-logits.jsonl');p=thinking/(role+'-logits.jsonl');raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest();c=[json.loads(x) for x in raw.splitlines()]
  approved={r['id']:r for r in splits[role]}
  for rows in [a,b,c]:
   if len(rows)!=count or len({r['id'] for r in rows})!=count or {r['id'] for r in rows}!=set(approved):raise ValueError('Incomplete or foreign role rows')
   for r in rows:
    if r['option_ids'][r['target_index']]!=approved[r['id']]['targets']['q'] or r['source']!=approved[r['id']]['source']:raise ValueError('Role target/source differs')
  inputs[role]=(a,b,c)
 value,cal,dev=run(*inputs['calibration'],*inputs['dev'])
 for p,expected in hashes.items():
  if digest(p)!=expected:raise ValueError('Thinking cal/dev changed during assessment')
 request={'commit':commit,'thinking_attempt':thinking.name,'input_sha256':hashes,'native_checksums_sha256':digest(native/'checksums.json'),'pool_checksums_sha256':digest(reverse/'checksums.json'),'card_sha256':hashlib.sha256(canonical(card).encode()).hexdigest(),'source_sha256':{n:digest(Path('/project')/n) for n in ['scripts/selective_thinking.py','scripts/assess_selective_thinking.py','scripts/modal_assess_selective_thinking.py']}}
 out=Path('/results/selective-order-thinking128-development-v1');prior=prepare_run(out,request)
 if prior is not None:return prior
 wb=None
 try:
  wb=wandb.init(project='ac-worktrial-decisions',name='selective-order-thinking128-v1',job_type='selective-thinking-dev',config=request)
  fitted=fit_temperature(cal);value.update(request=request,temperature=fitted,dev_raw=metrics(dev),dev_calibrated=metrics(dev,fitted['temperature']),dev_slices=slice_metrics(dev,fitted['temperature']),parameters_unchanged_final_verification_pending=True,wandb={'id':wb.id,'url':wb.url},GPU_allocated=False,optimizer_updates=0)
  for role,rows in [('calibration',cal),('dev',dev)]:
   (out/(role+'-logits.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows))
  wb.log({'dev_accuracy':value['gate']['selective_dev_accuracy'],'dev_gate':value['gate'],'dev_calibrated':value['dev_calibrated']});write_json(out/'result.json',value);seal(out);results.commit();return value
 finally:
  if wb is not None:wb.finish()
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();value=assess.remote(commit)
 name='selective-thinking-development.json' if value['status']!='waiting_for_complete_thinking_development' else 'selective-thinking-development-wait.json'
 (ROOT/'outputs/autoresearch'/name).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
