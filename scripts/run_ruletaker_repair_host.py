"""Future host repair driver; require completed/exported screen and released GPU."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';APP='ac-worktrial-ruletaker-repair-warm';DEADLINE=1792125869;IDLE=1200
NAMES=['ruletaker','aegis2','civilcomments','snli-multinli','massive','banking77']
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prerequisites(root,containers):
 root=Path(root);folder=root/'outputs/autoresearch'
 if json.loads((folder/'math-source-grid-status.json').read_text())['phase']!='complete':raise RuntimeError('Complete original source grid first')
 if not (folder/'math-worker-retired.json').exists():raise RuntimeError('Retire original host keepalive first')
 blocked={'ac-worktrial-math-warm-research','ac-worktrial-warm-research','ac-worktrial-pretrained-readout','ac-worktrial-head-determinism-probe','ac-worktrial-autoresearch-attention-repeats','ac-worktrial-readout-order-warm'}
 if any(x['app_name'] in blocked for x in containers):raise RuntimeError('Prior GPU worker still exists; no second GPU')
 for name in NAMES:
  run='ab1-math-accuracy-seed20261014-'+name;meta=root/'outputs/training'/run;expected=json.loads((meta/'checksums.json').read_text())['best-head.safetensors']
  candidates=[p for folder in (root/'outputs/model-artifacts').glob(run+'*') for p in folder.rglob('best-head.safetensors')]
  if not any(digest(p)==expected for p in candidates):raise RuntimeError('Verified selected tensor not exported: '+name)
  test=root/'outputs/test'/(run+'-test');checks=json.loads((test/'checksums.json').read_text())
  for filename in ['config.json','completion.json','metrics.json']:
   if digest(test/filename)!=checks[filename]:raise ValueError('Changed exported test metadata: '+name)
  if json.loads((test/'completion.json').read_text())['status']!='complete':raise RuntimeError('Incomplete source test: '+name)
 return True

def main():
 import modal
 from modal.config import _set_profile
 from durable_modal_calls import invoke
 _set_profile('ac-worktrial');folder=ROOT/'outputs/autoresearch'
 containers=json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal','container','list','--profile','ac-worktrial','--env',ENV,'--json'],text=True));prerequisites(ROOT,containers)
 proof=json.loads((folder/'ruletaker-cwa-worker-cpu.json').read_text());source=digest(ROOT/'scripts/modal_ruletaker_repair.py')
 if proof['status']!='passed' or source!=proof['worker_source_sha256']:raise RuntimeError('CPU-checked worker source changed')
 worker=modal.Cls.from_name(APP,'RepairWorker',environment_name=ENV)();state=worker.ping.remote()
 if state['failed'] or state['worker_source_sha256']!=source:raise RuntimeError('Worker failed or stale deployment')
 identity=state['worker_id'];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();run_id='ruletaker-cwa-explicit-math-seed20261014'
 request={'run_id':run_id,'commit':commit,'worker_id':identity,'worker_source_sha256':source}
 value=invoke(folder/'ruletaker-repair-call.json',request,lambda:worker.run.spawn(run_id,commit),modal.FunctionCall.from_id)
 if value['status']!='complete' or value['worker']['worker_id']!=identity:raise RuntimeError('Incomplete repair or changed worker')
 (folder/'ruletaker-repair-result.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
 while time.time()+IDLE+60<DEADLINE:
  for _ in range(30):
   if (folder/'repair-worker-retired.json').exists():return
   if time.time()+IDLE+60>=DEADLINE:return
   time.sleep(10)
  state=worker.ping.remote()
  if state['failed'] or state['worker_id']!=identity:raise RuntimeError('Repair worker changed or failed')
  (folder/'repair-worker-keepalive.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
