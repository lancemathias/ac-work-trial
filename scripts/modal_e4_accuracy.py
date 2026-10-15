"""Recover E4's best-accuracy saved head; no optimizer updates or backbone extraction."""
import json,time,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';PARENT='e4-qwen38-27b-v3-seed20261014-2d9ef9a'
app=modal.App('ac-worktrial-e4-accuracy-selection')
CACHE='/data/hidden-cache/e4-qwen38-27b-v3-batch4-ef15c46f4919'
DEADLINE=1792125869
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs')
 .add_local_dir(ROOT/'tests','/project/tests').add_local_dir(ROOT/'data/manifests','/project/data/manifests').add_local_dir(ROOT/'data/processed','/project/data/processed'))
data=modal.Volume.from_name('ac-worktrial-training-data',environment_name=ENV)
models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
def require_budget(seconds):
 if time.time()+seconds>DEADLINE:raise TimeoutError('Insufficient reserved A100 time')

@app.function(image=image,cpu=4,memory=8192,timeout=600,retries=0,scaledown_window=2)
def verify_cpu():
 import sys
 p=subprocess.run([sys.executable,'-m','unittest','-v','test_checkpoint_selection','test_training','test_cached_training','test_training_runtime','test_adamw_groups','test_trained_joint_evaluation.EvaluationTests'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-9000:])
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=98304,timeout=3600,retries=0,max_containers=1,scaledown_window=2,volumes={'/data':data,'/models':models,'/results':results},secrets=[secret])
def recover_and_evaluate(run_id,commit):
 require_budget(3600)
 import sys,gc,copy,torch,shutil
 sys.path.insert(0,'/project/scripts')
 from transformers import AutoProcessor
 from safetensors.torch import save_file
 from training_data import digest,load_splits
 from train_joint import prepare,predict,metrics,fit_temperature,slice_metrics,write_json,append_json
 from checkpoint_selection import best_epoch
 from frozen_feature_cache import FrozenFeatureBackbone
 from vendor.clef_joint_schema import JointSchemaHead
 results.reload();data.reload();parent=Path('/results')/PARENT;hashes=json.loads((parent/'checksums.json').read_text())
 for name in ['config.json','summary.json','metrics.jsonl','last-training-state.pt']:
  if digest(parent/name)!=hashes[name]:raise ValueError('Parent artifact changed: '+name)
 provenance=json.loads((parent/'config.json').read_text());cfg=copy.deepcopy(provenance['config']);cfg['selection_metric']='accuracy';cfg['run_name']='e4-accuracy-selected'
 events=[json.loads(x) for x in (parent/'metrics.jsonl').read_text().splitlines()];selected=best_epoch(events)
 saved=torch.load(parent/'last-training-state.pt',map_location='cpu',weights_only=True)
 if saved['epoch']!=selected['epoch'] or saved['step']!=selected['step']:raise ValueError('Highest accuracy epoch is not retained; do not substitute another epoch')
 if selected['epoch']!=5 or saved['config']!=provenance['config']:raise ValueError('Unexpected saved E4 state')
 out=Path('/results')/run_id;out.mkdir(exist_ok=False);wb=None;started=time.monotonic()
 try:
  splits_path=Path('/results')/(run_id+'-splits');shutil.copytree(Path('/results')/(PARENT+'-splits'),splits_path)
  splits,manifest=load_splits(splits_path/'manifest.json')
  processor=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True)
  prepared={role:prepare(rows,processor.tokenizer,cfg['max_input_tokens']) for role,rows in splits.items()}
  torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
  cm=json.loads((Path(CACHE)/'manifest.json').read_text())
  backbone=FrozenFeatureBackbone(CACHE,cfg,prepared,torch.device('cuda'),digest(Path(CACHE)/'manifest.json'),digest('/project/scripts/benchmark_training_data.py'),expected_source_config=cm['identity']['training_source_config'],expected_extraction_batch_size=4)
  head=JointSchemaHead(hidden_size=cm['identity']['hidden_size'],**cfg['head']).float().cuda();head.load_state_dict(saved['head']);head.requires_grad_(False).eval()
  save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(out/'best-head.safetensors'));del saved
  dev=predict(backbone,head,prepared['dev'],processor.tokenizer.pad_token_id,torch.device('cuda'),True);dm=metrics(dev)
  if dm['accuracy']!=selected['accuracy']:raise ValueError('Saved checkpoint dev accuracy failed reproduction')
  calibration=predict(backbone,head,prepared['calibration'],processor.tokenizer.pad_token_id,torch.device('cuda'),True);fitted=fit_temperature(calibration);backbone.verify_unchanged()
  origin={'parent':PARENT,'saved_state_sha256':hashes['last-training-state.pt'],'selection':'maximum dev accuracy; earliest epoch on ties','selected_epoch':selected['epoch'],'additional_optimizer_updates':0,'commit':commit}
  provenance.update(config=cfg,selection_origin=origin,selection_runner_sha256=digest(__file__));write_json(out/'config.json',provenance);write_json(out/'selection-origin.json',origin)
  write_json(out/'best.json',{'epoch':selected['epoch'],'step':selected['step'],'dev':dm,'selection_metric':'accuracy'});write_json(out/'temperature.json',fitted)
  for role,items in [('dev',dev),('calibration',calibration)]:
   for item in items:
    z=torch.tensor(item['logits'],dtype=torch.float64);append_json(out/(role+'-traces.jsonl'),dict(item,checkpoint_sha256=digest(out/'best-head.safetensors'),raw_probabilities=z.softmax(-1).tolist(),calibrated_probabilities=(z/fitted['temperature']).softmax(-1).tolist()))
  summary=json.loads((parent/'summary.json').read_text());summary.update(selection_origin=origin,selection_metric='accuracy',best_dev_nll=dm['nll'],dev_raw=dm,dev_calibrated=metrics(dev,fitted['temperature']),dev_raw_slices=slice_metrics(dev),dev_calibrated_slices=slice_metrics(dev,fitted['temperature']),temperature=fitted,calibration_raw=metrics(calibration),calibration_fitted=metrics(calibration,fitted['temperature']),reselection_seconds=time.monotonic()-started)
  write_json(out/'summary.json',summary)
  import wandb
  wb=wandb.init(project=cfg['wandb_project'],name=cfg['run_name'],config={'selection_origin':origin,'config':cfg});wb.summary.update(summary)
  for event in events:
   if event.get('event')=='dev_epoch':wb.log({'parent_dev_epoch/'+k:v for k,v in event.items() if k!='event'})
  write_json(out/'wandb-run.json',{'id':wb.id,'url':wb.url,'project':cfg['wandb_project']});wb.finish();wb=None
  write_json(out/'checksums.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='checksums.json'});results.commit()
  del backbone,head,prepared;gc.collect();torch.cuda.empty_cache()
  from evaluate_trained_joint import evaluate
  test=Path('/results')/(run_id+'-test');evaluate(out,test,commit,results.commit)
  from report_trained_joint import report
  result=report(test,label='E4 accuracy-selected Qwen3.8-27B');w=wandb.Api().run('lmathias/ac-worktrial-decisions/'+json.loads((out/'wandb-run.json').read_text())['id']);w.summary['test_calibrated']={k:v for k,v in result.items() if k!='suites'};w.summary['test_calibrated_per_suite']=result['suites'];w.summary.update()
  write_json(test/'pipeline.json',{'status':'complete','run_id':run_id,'selection_metric':'accuracy','commit':commit});write_json(test/'checksums.json',{p.name:digest(p) for p in test.iterdir() if p.is_file() and p.name!='checksums.json'});return {'status':'complete','composite':result['macro_coverage_adjusted_accuracy']}
 except Exception as exc:
  write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:
  if wb is not None:wb.finish()
  results.commit();models.commit()
@app.function(image=image,cpu=.25,memory=512,timeout=7200,retries=0,volumes={'/results':results})
def pipeline(run_id,commit):
 out=Path('/results')/(run_id+'-pipeline');out.mkdir(exist_ok=False)
 def state(phase,**kw):
  (out/'status.json').write_text(json.dumps({'phase':phase,'run_id':run_id,'commit':commit,**kw},indent=2)+'\n');results.commit()
 try:
  state('cpu_validation');validation=verify_cpu.remote();(out/'cpu-validation.json').write_text(json.dumps(validation,indent=2)+'\n');results.commit();state('recovering_accuracy_head_and_evaluating');result=recover_and_evaluate.remote(run_id,commit);state('complete',result=result);return result
 except Exception as exc:state('failed',error=type(exc).__name__,message=str(exc));raise
@app.local_entrypoint()
def main(run_id:str,cpu_only:bool=False):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 if cpu_only:
  r=verify_cpu.remote();(ROOT/'outputs/autoresearch/accuracy-selection-cpu-tests.json').write_text(json.dumps(r,indent=2)+'\n');print(r['stderr'][-4000:]);return
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();print(json.dumps(pipeline.remote(run_id,commit),indent=2))
