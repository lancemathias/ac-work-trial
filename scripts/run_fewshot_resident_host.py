"""Run only after checked deployment and retirement of the prior single GPU worker."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';APP='ac-worktrial-readout-candidates-warm';DEADLINE=1792125869;IDLE=1200

def main():
 import modal
 from modal.config import _set_profile
 from durable_modal_calls import invoke
 _set_profile('ac-worktrial');folder=ROOT/'outputs/autoresearch'
 proof=json.loads((folder/'readout-candidates-worker-cpu.json').read_text());sha=hashlib.sha256((ROOT/'scripts/modal_readout_candidates.py').read_bytes()).hexdigest()
 if proof['status']!='passed' or proof['worker_source_sha256']!=sha:raise RuntimeError('Exact CPU-tested candidate service required')
 # This host never stops the active thinking run. Handoff is an explicit step.
 if not (folder/'staged-readout-worker-retired.json').exists():raise RuntimeError('Retire current thinking host and release its GPU before handoff')
 launch=json.loads((folder/'readout-candidates-launch.json').read_text())
 if launch['worker_source_sha256']!=sha or launch['phase']!='deployed_zero_idle':raise RuntimeError('Verified next deployment required')
 containers=json.loads(subprocess.check_output(['/Users/bwt/ac-worktrial/.venv/bin/modal','container','list','--profile','ac-worktrial','--env',ENV,'--json'],text=True))
 if containers:raise RuntimeError('Verify prior containers are released before the single-A100 handoff')
 worker=modal.Cls.from_name(APP,'ReadoutCandidates',environment_name=ENV)();state=worker.ping.remote();identity=state['worker_id']
 if state['failed'] or state['source_sha256']!=sha:raise RuntimeError('Stale or failed resident worker')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 def call(key,run_id,spawn):
  request=dict(run_id=run_id,commit=commit,worker_id=identity,worker_source_sha256=sha)
  value=invoke(folder/(key+'-call.json'),request,spawn,modal.FunctionCall.from_id)
  if value['worker_id']!=identity:raise RuntimeError('Worker changed; diagnose before further inference')
  (folder/(key+'-result.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True);return value
 for label in ['cold','warm']:
  name='readout-fewshot-quality-v2-'+label+'-smoke'
  value=call('fewshot-quality-v2-'+label,name,lambda:worker.fewshot_smoke.spawn(name))
  if value['status']!='complete' or not value['capacity_validated']:raise RuntimeError('Fewshot capacity failed; no accuracy trial')
 name='readout-native-seven-quality-v2'
 value=call('fewshot-quality-v2-trial',name,lambda:worker.fewshot_trial.spawn(name,commit))
 if value['status'] not in ['complete','rejected_on_dev','held_for_development_budget','held_for_test_budget']:raise RuntimeError('Unexpected fewshot outcome')
 while time.time()+IDLE+60<DEADLINE:
  for _ in range(30):
   if (folder/'candidate-readout-worker-retired.json').exists() or time.time()+IDLE+60>=DEADLINE:return
   time.sleep(10)
  state=worker.ping.remote()
  if state['failed'] or state['worker_id']!=identity:raise RuntimeError('Worker changed or failed; stop keepalive')
  (folder/'candidate-readout-keepalive.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
