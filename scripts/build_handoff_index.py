"""Build a handoff index from saved evidence; never select a new checkpoint."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(root,audit):
 root,audit=Path(root),Path(audit);remote=read(audit/'modal-inventory.json',{});remote_files={r['path']:r for r in remote.get('ac-worktrial-training-results',[]) if r['type']=='1'}
 wb=read(audit/'wandb-runs.json',{'edges':[]});wbrows=[e['node'] for e in wb['edges']]
 previous=read(root/'outputs/handoff/run-index.json',{});saved={r['run_id']:r for r in previous.get('trained_runs',[])}
 entries=[]
 for p in sorted((root/'outputs/training').iterdir()):
  cfg=read(p/'config.json',{});summary=read(p/'summary.json',{});best=read(p/'best.json');checks=read(p/'checksums.json',{})
  if not cfg:continue
  settings=cfg.get('config',cfg)
  entry={'run_id':p.name,'kind':'trained_head','status':summary.get('status','incomplete_or_historical'),'config':str((p/'config.json').relative_to(root)),'config_sha256':sha(p/'config.json'),'backbone_id':settings.get('backbone_id'),'backbone_revision':settings.get('backbone_revision'),'seed':settings.get('seed'),'head':settings.get('head'),'selection':(best or {}).get('selection_metric',settings.get('selection_metric','nll')),'accepted_epoch':(best or {}).get('epoch'),'accepted_step':(best or {}).get('step'),'autoresearch':p.name.startswith(('e4-fast','e5','ab1','ruletaker')),'wandb':read(p/'wandb-run.json',{})}
  candidates=[k for k in remote_files if k.startswith(p.name+'/') and k.endswith('/best-head.safetensors')]
  if p.name.endswith('-interrupted-attempt'):candidates=[]
  expected=checks.get('best-head.safetensors')
  if candidates and best:
   # Multi-attempt runs retain every attempt. Only a matching local accepted hash is safe to advertise.
   entry['accepted_checkpoint']={'volume':'ac-worktrial-training-results','environment':'worktrial-lance','path':candidates[0] if len(candidates)==1 else None,'sha256':expected,'candidate_paths':candidates if len(candidates)>1 else [],'verification':'pending_remote_hash_check'}
   entry['other_checkpoints']={'volume':'ac-worktrial-training-results','environment':'worktrial-lance','directory':p.name,'policy':'Retained and reachable; not individually referenced as accepted.'}
  else:entry['accepted_checkpoint']=None
  old=saved.pop(p.name,None)
  if old and old.get('config_sha256')==entry.get('config_sha256') and old.get('accepted_checkpoint',{}):
   entry['accepted_checkpoint']=old['accepted_checkpoint']
  entries.append(entry)
 entries.extend(saved.values())
 scored=read(root/'outputs/autoresearch/overnight-benchmark-scores.json',{})
 branches=[]
 for line in subprocess.check_output(['git','-C',str(root),'for-each-ref','refs/heads','--format=%(refname:short) %(objectname)'],text=True).splitlines():
  name,commit=line.split();branches.append({'branch':name,'commit':commit,'role':'autoresearch' if name.startswith('autoresearch/') else 'preservation' if name.startswith('preservation/') else 'handoff' if name.startswith('deliverables/') else 'historical'})
 index={'policy':'Reference only the accepted checkpoint per run; other saved checkpoints remain reachable in their original run directory. No selection is performed by this index.','workspace':'ac-worktrials','profile':'ac-worktrial','environment':'worktrial-lance','wandb_project':'https://wandb.ai/lmathias/ac-worktrial-decisions','model_pins':{'Qwen/Qwen3.5-9B':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','Qwen/Qwen3.8-27B':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','Cloudflare/clef_head_source':'0b331204bb13fbd2ca93a64df1956f5b55478ce5'},'trained_runs':entries,'completed_benchmark_scores':scored.get('scores',{}),'wandb_runs':[{'id':r['name'],'name':r['displayName'],'state_at_audit':r['state'],'url':'https://wandb.ai/lmathias/ac-worktrial-decisions/runs/'+r['name']} for r in wbrows],'branches':branches,'preservation':read(audit/'preservation.json',[])}
 aliases=read(root/'configs/resource-aliases.json',{})
 experiment=aliases.get('experiment',{})
 for entry in index['trained_runs']:
  if entry['run_id']==experiment.get('legacy_id'):
   entry['display_name']=experiment['display_name'];entry['canonical_id']=experiment['canonical_id']
 names={x['id']:x['display_name'] for x in aliases.get('wandb_runs',[])}
 for row in index['wandb_runs']:
  if row['id'] in names:row['name']=names[row['id']]
 dest=root/'outputs/handoff/run-index.json';dest.write_text(json.dumps(index,indent=2)+'\n');print('Indexed',len(entries),'training records,',len(index['completed_benchmark_scores']),'scored conditions,',len(wbrows),'W&B runs')
 return index
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--audit',default='work/deliverable-audit');a=p.parse_args();build(Path(a.root).resolve(),a.audit)
