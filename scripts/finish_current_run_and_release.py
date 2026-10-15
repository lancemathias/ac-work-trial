"""Explicit user-authorized closeout: wait for one existing call, export, stop its app."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ENV='worktrial-lance';APP_ID='ap-gPxM8tSU0bSbT5Pr7mJ14M';RUN='readout-bounded-thinking128-staged-v2';CALL='fc-01M5055P0HY6PEK6ERTJ4RY839';MODAL='/Users/bwt/ac-worktrial/.venv/bin/modal'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,value):
 path=ROOT/'outputs/autoresearch'/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
 import modal
 from modal.config import _set_profile
 from export_training import export
 _set_profile('ac-worktrial');folder=ROOT/'outputs/autoresearch';policy=json.loads((folder/'research-stop-after-current.json').read_text());launch=json.loads((folder/'staged-readout-launch.json').read_text())
 if (policy['last_allowed_run'],policy['function_call_id'],policy['modal_app'])!=(RUN,CALL,APP_ID) or launch['modal_app']!=APP_ID:raise ValueError('Closeout must match explicit user instruction and current deployment')
 if not (folder/'staged-readout-worker-retired.json').exists():raise RuntimeError('Disable post-run keepalive before closeout')
 completed=folder/'gpu-release-receipt.json'
 if completed.exists() and json.loads(completed.read_text()).get('GPU_released'):print('Existing verified GPU release receipt; no replay',flush=True);return
 call=modal.FunctionCall.from_id(CALL);terminal=None;failure=None
 while True:
  try:terminal=call.get(timeout=45);break
  except TimeoutError:
   state={'phase':'waiting_only_for_current_run','checked_utc':now(),'function_call_id':CALL,'modal_app':APP_ID,'new_GPU_jobs_allowed':False};write('gpu-release-status.json',state);print(json.dumps(state),flush=True)
  except Exception as exc:
   write('gpu-release-status.json',{'phase':'observation_error_needs_terminal_verification','checked_utc':now(),'error_type':type(exc).__name__,'modal_app':APP_ID,'qualification':'No export or stop attempted: an observation error alone does not establish terminal GPU work.'})
   raise
 write('gpu-release-status.json',{'phase':'current_call_terminal_exporting','checked_utc':now(),'result':terminal,'terminal_error_type':failure,'modal_app':APP_ID})
 destination=ROOT/'outputs/model-artifacts'/RUN;export_error=None
 try:
  if destination.exists():
   checks=json.loads((destination/'checksums.json').read_text())
   if not checks or any(hashlib.sha256((destination/n).read_bytes()).hexdigest()!=h for n,h in checks.items()):raise ValueError('Existing local export did not verify')
  else:export(RUN,destination)
  write('thinking-final-export.json',{'status':'verified','local_root':str(destination),'checksums_sha256':hashlib.sha256((destination/'checksums.json').read_bytes()).hexdigest(),'run_id':RUN,'exported_utc':now(),'Modal_volumes_retained':True})
 except Exception as exc:
  # Stopping a compute app leaves persistent results/model volumes intact.
  # Do not hold the user's GPU indefinitely because an export transfer failed.
  export_error=type(exc).__name__;write('thinking-export-pending.json',{'status':'export_pending_persistent_volume_retained','error_type':export_error,'run_id':RUN,'time':now(),'qualification':'No persistent volume is removed; recover existing committed results on CPU.'})
 flags=['--profile','ac-worktrial','--env',ENV]
 stopped=subprocess.run([MODAL,'app','stop',APP_ID,'--yes',*flags],capture_output=True,text=True,timeout=60)
 write('gpu-release-status.json',{'phase':'stop_requested_verifying_container_release','checked_utc':now(),'modal_app':APP_ID,'stop_returncode':stopped.returncode,'export_error_type':export_error})
 released=False;remaining=[];app_state=None
 for observation in range(12):
  containers=json.loads(subprocess.check_output([MODAL,'container','list',*flags,'--json'],text=True,timeout=30));apps=json.loads(subprocess.check_output([MODAL,'app','list',*flags,'--json'],text=True,timeout=30))
  remaining=[c for c in containers if c.get('app_id')==APP_ID or c.get('app_name')=='ac-worktrial-readout-order-warm'];app_state=next((a['state'] for a in apps if a['app_id']==APP_ID),None)
  if not remaining and app_state=='stopped':released=True;break
  time.sleep(5)
 receipt={'GPU_released':released,'checked_utc':now(),'modal_app':APP_ID,'app_state':app_state,'remaining_target_containers':remaining,'stop_returncode':stopped.returncode,'terminal_result':terminal,'terminal_error_type':failure,'export_error_type':export_error,'persistent_volumes_retained':True,'new_GPU_jobs_allowed':False}
 write('gpu-release-receipt.json',receipt);print(json.dumps(receipt),flush=True)
 if not released:raise RuntimeError('Stop requested but target GPU release not yet verified; inspect same app, never launch a new GPU job')
if __name__=='__main__':main()
