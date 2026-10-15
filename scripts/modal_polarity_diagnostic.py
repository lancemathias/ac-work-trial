"""Validation-only polarity probe after the locked source-ablation models finish."""
import json,time,subprocess
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';DEADLINE=1792125869;BATCH='ab1-accuracy-seed20261014'
NAMES=['existing-source-control','ruletaker','aegis2','civilcomments','snli-multinli','massive','banking77']
RUNS=['e3-medium-v3-seed20261014-e7dd4bb']+[BATCH+'-'+x for x in NAMES]
app=modal.App('ac-worktrial-validation-polarity-probe')
image=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(ROOT/'requirements-training.txt')
 .add_local_dir(ROOT/'scripts','/project/scripts').add_local_dir(ROOT/'configs','/project/configs').add_local_dir(ROOT/'tests','/project/tests')
 .add_local_file(ROOT/'data/training/polarity-dev-probe.jsonl','/project/polarity-probe.jsonl')
 .add_local_file(ROOT/'outputs/model-artifacts/e3-medium-v3-seed20261014-e7dd4bb-splits/dev.jsonl','/project/source-dev.jsonl')
 .add_local_file(ROOT/'outputs/training/e3-medium-v3-seed20261014-e7dd4bb/config.json','/project/e3-provenance.json'))
results=modal.Volume.from_name('ac-worktrial-training-results',environment_name=ENV);models=modal.Volume.from_name('ac-worktrial-models',environment_name=ENV)
secret=modal.Secret.from_name('ac-worktrial-wandb',environment_name=ENV,required_keys=['WANDB_API_KEY'])
def budget():
 if time.time()+1800>DEADLINE:raise TimeoutError('Outside bounded reserved A100 window')
@app.function(image=image,cpu=1,memory=2048,timeout=300,retries=0,scaledown_window=2)
def verify_probe():
 import sys
 sys.path.insert(0,'/project/scripts')
 from training_data import digest
 from polarity_diagnostic import pairs
 from jsonschema import Draft202012Validator
 provenance=json.loads(Path('/project/e3-provenance.json').read_text());source=Path('/project/source-dev.jsonl')
 if digest(source)!=provenance['manifest']['splits']['dev']['points']['sha256']:raise ValueError('Dev data changed')
 rows=[json.loads(x) for x in source.open()];probe=[json.loads(x) for x in Path('/project/polarity-probe.jsonl').open()]
 if pairs(rows)!=probe or len(probe)!=188:raise ValueError('Invalid paired diagnostic')
 validator=Draft202012Validator(json.loads(Path('/project/configs/normalized-training.schema.json').read_text()))
 if any(validator.is_valid(r) for r in probe):raise ValueError('Diagnostic rows must be rejected by training schema')
 p=subprocess.run([sys.executable,'-m','unittest','-v','test_polarity_diagnostic'],cwd='/project/tests',capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return {'status':'passed','pairs':94,'source_dev_sha256':digest(source),'probe_sha256':digest('/project/polarity-probe.jsonl'),'training_schema_rejects_all_probe_rows':True,'tests':p.stderr}
@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=1800,max_containers=1,retries=0,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results},secrets=[secret])
def evaluate_heads(probe_id,commit):
 budget()
 import gc,sys,torch
 sys.path.insert(0,'/project/scripts')
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from safetensors.torch import load_file
 from vendor.clef_joint_schema import JointSchemaHead,encode_record,collate_records
 from train_joint import write_json
 from training_data import digest
 from polarity_diagnostic import metrics
 cfg=json.loads(Path('/project/configs/e3-medium-v3.json').read_text());out=Path('/results')/probe_id;out.mkdir(exist_ok=False);summary={}
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
 try:
  processor=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True)
  backbone=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'}).requires_grad_(False).eval()
  frozen=[(p,p._version) for p in backbone.parameters()];text=backbone.model
  if hasattr(text,'language_model'):text=text.language_model
  rows=[json.loads(x) for x in Path('/project/polarity-probe.jsonl').open()];prepared=[]
  for r in rows:
   encoded=encode_record(processor.tokenizer,{'id':r['id'],'state':r['state'],'questions':{'q':r['question']}},max_length=10**9)
   if len(encoded.input_ids)>8192:raise ValueError('Probe overlength; no truncation')
   prepared.append((r,encoded))
  prepared.sort(key=lambda x:(len(x[1].input_ids),x[0]['id']));cache=[]
  with torch.no_grad():
   for start in range(0,len(prepared),4):
    group=prepared[start:start+4];batch=collate_records([e for _,e in group],processor.tokenizer.pad_token_id,torch.device('cuda'))
    h=text(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
    for i,(r,e) in enumerate(group):cache.append((r,e,h[i,:len(e.input_ids)].cpu().contiguous()))
   del h,batch
   for run_id in RUNS:
    training=Path('/results')/run_id;checks=json.loads((training/'checksums.json').read_text())
    for name in ['config.json','best-head.safetensors','temperature.json']:
     if digest(training/name)!=checks[name]:raise ValueError('Locked model artifact changed')
    model_cfg=json.loads((training/'config.json').read_text())['config'];temperature=json.loads((training/'temperature.json').read_text())['temperature']
    if any(model_cfg[k]!=cfg[k] for k in ['backbone_id','backbone_revision','head']):raise ValueError('Nonmatched probe model')
    if model_cfg.get('attention_mode','native_causal')!='native_causal':raise ValueError('Probe features incompatible with model attention')
    head=JointSchemaHead(hidden_size=backbone.get_output_embeddings().weight.shape[1],**model_cfg['head']).float().cuda();head.load_state_dict(load_file(training/'best-head.safetensors',device='cuda'));head.requires_grad_(False).eval();head_versions=[(p,p._version) for p in head.parameters()];predictions=[]
    for r,e,hidden in cache:
     b=collate_records([e],processor.tokenizer.pad_token_id,torch.device('cuda'))
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=head(hidden.unsqueeze(0).cuda(),b['input_ids'],b['attention_mask'],b['records'],backbone.get_output_embeddings().weight.detach())[0][0].float()
     p_true=float((logits/temperature).softmax(-1)[e.questions[0].option_ids.index('true')]);predictions.append({k:r[k] for k in ['id','diagnostic_parent_id','polarity_variant','label']}|{'p_true':p_true})
    if any(p.requires_grad or p.grad is not None or p._version!=version for p,version in head_versions):raise RuntimeError('Probe changed head')
    summary[run_id]=dict(metrics(predictions),temperature=temperature,checkpoint_sha256=checks['best-head.safetensors'])
    (out/(run_id+'-predictions.jsonl')).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in predictions));write_json(out/'summary.json',summary);results.commit()
    try:
     import wandb
     wb=json.loads((training/'wandb-run.json').read_text());w=wandb.Api().run('lmathias/ac-worktrial-decisions/'+wb['id']);w.summary['dev_polarity_probe']=summary[run_id];w.summary.update()
    except Exception as exc:write_json(out/(run_id+'-wandb-warning.json'),{'error':type(exc).__name__})
    del head;gc.collect();torch.cuda.empty_cache()
  if any(p.requires_grad or p.grad is not None or p._version!=version for p,version in frozen):raise RuntimeError('Probe changed backbone')
  write_json(out/'completion.json',{'status':'complete','commit':commit,'pairs':94,'models':len(RUNS),'all_parameters_unchanged':True,'role':'development diagnostic; no fitting or checkpoint selection','probe_sha256':digest('/project/polarity-probe.jsonl')});return {'status':'complete','summary':summary}
 except Exception as exc:
  write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)});raise
 finally:
  write_json(out/'checksums.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='checksums.json'});results.commit()
@app.function(image=image,cpu=.25,memory=512,timeout=12*3600,retries=0,max_containers=1,volumes={'/results':results})
def pipeline(probe_id,commit):
 out=Path('/results')/(probe_id+'-pipeline');out.mkdir(exist_ok=False)
 def status(phase,**kw):
  (out/'status.json').write_text(json.dumps({'phase':phase,'probe_id':probe_id,'commit':commit,**kw},indent=2)+'\n');results.commit()
 try:
  status('verifying_probe');check=verify_probe.remote();(out/'validation.json').write_text(json.dumps(check,indent=2)+'\n');results.commit();status('waiting_for_source_grid')
  while True:
   budget();results.reload();p=Path('/results')/(BATCH+'-pipeline')/'status.json'
   if p.exists():
    state=json.loads(p.read_text())['phase']
    if state=='complete':break
    if state=='failed':raise RuntimeError('Source grid failed; halt probe for diagnosis')
   time.sleep(30)
  status('evaluating_locked_heads');result=evaluate_heads.remote(probe_id,commit);status('complete');return result
 except Exception as exc:status('failed',error=type(exc).__name__,message=str(exc));return {'status':'failed','error':str(exc)}
@app.local_entrypoint()
def main(probe_id:str):
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();print(json.dumps(pipeline.remote(probe_id,commit),indent=2))

@app.local_entrypoint()
def check():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=verify_probe.remote();(ROOT/'outputs/autoresearch/polarity-data-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
