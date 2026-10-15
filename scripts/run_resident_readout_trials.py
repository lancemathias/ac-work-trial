"""Host-only future27B driver. Never deploys, stops or allocates a second worker."""
import json,hashlib,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';APP='ac-worktrial-readout-order-warm';DEADLINE=1792125869;IDLE=1200

def main():
 import modal
 from modal.config import _set_profile
 from durable_modal_calls import invoke
 _set_profile('ac-worktrial');folder=ROOT/'outputs/autoresearch'
 grid=json.loads((folder/'math-source-grid-status.json').read_text())
 if grid['phase']!='complete':raise RuntimeError('Finish original source grid before GPU handoff')
 if not (folder/'math-worker-retired.json').exists():raise RuntimeError('Retire original host keepalive before GPU handoff')
 containers=json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal','container','list','--profile','ac-worktrial','--env',ENV,'--json'],text=True))
 blocked={'ac-worktrial-math-warm-research','ac-worktrial-warm-research','ac-worktrial-pretrained-readout','ac-worktrial-head-determinism-probe','ac-worktrial-autoresearch-attention-repeats'}
 if any(x['app_name'] in blocked for x in containers):raise RuntimeError('Old GPU worker remains; do not allocate another')
 worker=modal.Cls.from_name(APP,'ReadoutWorker',environment_name=ENV)()
 state=worker.ping.remote();expected=hashlib.sha256((ROOT/'scripts/modal_readout_order.py').read_bytes()).hexdigest()
 if state['failed'] or state.get('worker_source_sha256')!=expected:raise RuntimeError('Worker failed or deployment source differs')
 identity=state['worker_id'];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 def run(key,run_id,spawn):
  request={'run_id':run_id,'commit':commit,'worker_id':identity,'worker_source_sha256':expected}
  value=invoke(folder/(key+'-call.json'),request,spawn,modal.FunctionCall.from_id)
  if value['worker_id']!=identity:raise RuntimeError('Worker changed; retain evidence and diagnose')
  (folder/(key+'-result.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True);return value
 order_id='readout-native-reverse-pool-v1'
 order=run('resident-order',order_id,lambda:worker.trial.spawn(order_id,commit))
 if order['status'] not in ['complete','rejected_on_dev']:raise RuntimeError('Order trial incomplete')
 smoke_id='readout-thinking128-smoke-v1'
 smoke=run('resident-thinking-smoke',smoke_id,lambda:worker.thinking_smoke.spawn(smoke_id))
 if smoke['status']=='complete' and smoke.get('capacity_validated') and smoke['capacity_gate']['status']=='passed':
  run_id='readout-bounded-thinking128-v1'
  value=run('resident-thinking',run_id,lambda:worker.thinking_trial.spawn(run_id,commit))
  if value['status'] not in ['complete','rejected_on_dev']:raise RuntimeError('Thinking trial incomplete')
 # Continue retaining the loaded model for further authorized research.
 while time.time()+IDLE+60<DEADLINE:
  for _ in range(30):
   if (folder/'readout-worker-retired.json').exists():return
   if time.time()+IDLE+60>=DEADLINE:return
   time.sleep(10)
  value=worker.ping.remote()
  if value['failed'] or value['worker_id']!=identity:raise RuntimeError('Resident worker changed or failed')
  (folder/'readout-worker-keepalive.json').write_text(json.dumps(value,indent=2)+'\n')
if __name__=='__main__':main()
