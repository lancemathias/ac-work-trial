"""Host keepalive for the already confirmed worker; bounded by reservation/retirement."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEADLINE=1792125869;IDLE=1200

def main():
 import modal
 from modal.config import _set_profile
 _set_profile('ac-worktrial');folder=ROOT/'outputs/autoresearch';proof=json.loads((folder/'math-source-full-repeat-proof.json').read_text());expected=proof['first']['worker']['worker_id'];out=folder/'math-worker-host-keepalive.json';retired=folder/'math-worker-retired.json'
 worker=modal.Cls.from_name('ac-worktrial-math-warm-research','MathWorker',environment_name='worktrial-lance')()
 def status(phase,**kw):out.write_text(json.dumps(dict(phase=phase,time=time.time(),worker_id=expected,**kw),indent=2)+'\n')
 try:
  while time.time()+IDLE+60<DEADLINE:
   if retired.exists():status('retired');return
   status('ping_requested');state=worker.ping.remote()
   if state['failed'] or state['worker_id']!=expected:raise RuntimeError('Confirmed worker changed or failed; stop keepalive for diagnosis')
   status('warm',worker=state)
   for _ in range(30):
    if retired.exists():status('retired');return
    if time.time()+IDLE+60>=DEADLINE:break
    time.sleep(10)
  status('reservation_expiry')
 except Exception as exc:status('failed',error=type(exc).__name__,message=str(exc));raise
if __name__=='__main__':main()
