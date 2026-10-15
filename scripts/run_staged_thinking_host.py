"""Fresh label-blind cold/warm smoke, then independently budgeted thinking phases."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';APP='ac-worktrial-readout-order-warm';DEADLINE=1792125869;IDLE=1200

def main():
 import modal
 from modal.config import _set_profile
 from durable_modal_calls import invoke
 _set_profile('ac-worktrial');folder=ROOT/'outputs/autoresearch'
 if not (folder/'readout-worker-retired.json').exists():raise RuntimeError('Retire original readout host keepalive before deployment handoff')
 proof=json.loads((folder/'readout-order-worker-cpu.json').read_text());sha=hashlib.sha256((ROOT/'scripts/modal_readout_order.py').read_bytes()).hexdigest()
 if proof['status']!='passed' or proof.get('worker_source_sha256')!=sha:raise RuntimeError('CPU-tested worker source required')
 containers=json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal','container','list','--profile','ac-worktrial','--env',ENV,'--json'],text=True))
 if containers:raise RuntimeError('Prior containers remain; verify single-GPU release before requesting worker')
 worker=modal.Cls.from_name(APP,'ReadoutWorker',environment_name=ENV)();state=worker.ping.remote()
 if state['failed'] or state['worker_source_sha256']!=sha:raise RuntimeError('Worker failed or stale deployment')
 identity=state['worker_id'];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 def call(key,run_id,spawn):
  request={'run_id':run_id,'commit':commit,'worker_id':identity,'worker_source_sha256':sha}
  value=invoke(folder/(key+'-call.json'),request,spawn,modal.FunctionCall.from_id)
  if value['worker_id']!=identity:raise RuntimeError('Worker changed; diagnose before new work')
  (folder/(key+'-result.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True);return value
 for label in ['cold','warm']:
  name='readout-thinking128-'+label+'-staged-v2'
  value=call('thinking-staged-'+label,name,lambda:worker.thinking_smoke.spawn(name))
  if value['status']!='complete' or not value.get('capacity_validated'):raise RuntimeError('Thinking capacity failed; no inference trial')
 name='readout-bounded-thinking128-staged-v2'
 value=call('thinking-staged-trial',name,lambda:worker.thinking_trial.spawn(name,commit))
 if value['status'] not in ['complete','rejected_on_dev','held_for_test_budget','held_for_development_budget']:raise RuntimeError('Unexpected thinking outcome')
 while time.time()+IDLE+60<DEADLINE:
  for _ in range(30):
   if (folder/'staged-readout-worker-retired.json').exists():return
   if time.time()+IDLE+60>=DEADLINE:return
   time.sleep(10)
  state=worker.ping.remote()
  if state['failed'] or state['worker_id']!=identity:raise RuntimeError('Worker changed or failed; halt keepalive')
  (folder/'staged-readout-keepalive.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
