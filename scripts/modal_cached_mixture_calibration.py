"""CPU-only calibration and W&B logging for the dev-qualified cached mixture."""
import json,sys,subprocess
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image,results,secret
app=modal.App('ac-worktrial-cached-mixture-calibration')
RUN='cached-e5-readout-equal-mixture-v1'
@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2,volumes={'/results':results},secrets=[secret])
def calibrate(gate,commit):
 from evaluate_cached_mixture import mixture,verified_rows,verified_json
 from training_data import load_splits,digest
 from train_joint import fit_temperature,metrics
 from head_probe_io import prepare_run,write_json,seal
 import wandb,collections
 if gate['status']!='dev_gate_passed' or gate['gate']['gain_vs_stronger_component']<.01-1e-12:raise ValueError('Passed preregistered dev gate required')
 head=Path('/results/e5-fast-accuracy-bidir-v3-seed20261014');lm=Path('/results/readout-fast-qwen38-27b-v1');out=Path('/results')/(RUN+'-calibration')
 expected=gate['components'];hc=verified_json(head,'config.json');lc=verified_json(lm,'config.json');lt=verified_json(lm,'temperature.json')['temperature']
 if lt!=expected['readout_temperature'] or lc['checkpoint_sha256']!=expected['readout_checkpoint_sha256']:raise ValueError('Readout parent differs')
 for role,key in [('dev','dev_sha256'),('calibration','calibration_sha256')]:
  if hc['manifest']['splits'][role]['points']['sha256']!=expected[key] or lc['manifest']['splits'][role]['points']['sha256']!=expected[key]:raise ValueError('Parent split identity differs')
 request={'commit':commit,'dev_gate':gate,'weights':[.5,.5],'head':'e5-fast-accuracy-bidir-v3-seed20261014','readout':'readout-fast-qwen38-27b-v1','runner_sha256':digest(__file__),'mixture_sha256':digest('/project/scripts/evaluate_cached_mixture.py')}
 prior=prepare_run(out,request);results.commit()
 if prior is not None:return prior
 splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json');cal=splits['calibration'];targets={r['id']:r['targets']['q'] for r in cal}
 a=verified_rows(head,'calibration-traces.jsonl');b=verified_rows(lm,'calibration-logits.jsonl');items=mixture(a,b,lt)
 if len(items)!=857 or set(targets)!={r['id'] for r in items}:raise ValueError('Calibration ID/role mismatch')
 if any(r['checkpoint_sha256']!=expected['head_checkpoint_sha256'] for r in a):raise ValueError('Head parent checkpoint differs')
 for r in items:
  if r['option_ids'][r['target_index']]!=targets[r['id']]:raise ValueError('Calibration label mismatch')
 fitted=fit_temperature(items);wb=None
 try:
  wb=wandb.init(project='ac-worktrial-decisions',name=RUN,job_type='cached-probability-mixture',config=request)
  wb.log({'dev_accuracy':gate['gate']['mixture_accuracy'],'calibration_raw':metrics(items),'calibration_fitted':metrics(items,fitted['temperature'])})
  value={'status':'complete','request':request,'temperature':fitted,'calibration_sources':dict(collections.Counter(r['source'] for r in items)),'calibration_decisions':857,'test_read_or_scored':False,'wandb':{'id':wb.id,'url':wb.url},'optimizer_updates':0,'GPU_forward_passes':0}
  write_json(out/'result.json',value);seal(out);results.commit();return value
 finally:
  if wb is not None:wb.finish()
@app.function(image=image,cpu=1,memory=2048,timeout=180,retries=0,scaledown_window=2,secrets=[secret])
def publish_test(run_id,metrics):
 import wandb
 run=wandb.Api().run('lmathias/ac-worktrial-decisions/'+run_id);run.summary['test_composite']=metrics['macro_coverage_adjusted_accuracy'];run.summary['test_per_suite']=metrics['suites'];run.summary.update();return {'status':'logged','run_id':run_id}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 gate=json.loads((ROOT/'outputs/autoresearch/cached-e5-readout-mixture-dev.json').read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 result=calibrate.remote(gate,commit);(ROOT/'outputs/autoresearch/cached-e5-readout-mixture-calibration.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
@app.local_entrypoint()
def publish():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 fit=json.loads((ROOT/'outputs/autoresearch/cached-e5-readout-mixture-calibration.json').read_text());m=json.loads((ROOT/'outputs/test'/RUN/'metrics.json').read_text());print(json.dumps(publish_test.remote(fit['wandb']['id'],m)))
