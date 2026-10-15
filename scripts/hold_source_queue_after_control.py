"""Intentional safe-boundary hold after the measured repeatability failure."""
import datetime,json,subprocess,time
from pathlib import Path
import modal
from modal.config import _set_profile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/autoresearch'
_set_profile('ac-worktrial')
volume=modal.Volume.from_name('ac-worktrial-training-results',environment_name='worktrial-lance')
receipt=OUT/'source-reproducibility-hold.json'
if receipt.exists():raise FileExistsError('Preserve existing handoff receipt')
def write(phase,**kwargs):
 row=dict(phase=phase,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Measured same-initial-head/same-cache/same-window training divergence; preserve current completed control, then validate deterministic head backward',**kwargs)
 receipt.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
write('waiting_for_current_control_completion')
while True:
 try:
  raw=b''.join(volume.read_file('warm-worker-jobs/warm-existing-source-control-v1.json'))
 except (FileNotFoundError,modal.exception.NotFoundError):
  time.sleep(5);continue
 except Exception as exc:
  # Transient observation errors never cause a stop or a relaunch.
  print(json.dumps({'event':'observation_error','type':type(exc).__name__}),flush=True);time.sleep(20);continue
 current=json.loads(raw)
 if current['status']!='complete':
  write('control_failed_requires_diagnosis',current=current);break
 (OUT/'official-control-before-hold-receipt.json').write_bytes(raw)
 stops=[]
 for app_id in ['ap-3tDGWsFm6ZdCIh4XjJOeF7','ap-iPVnuNNE1KPK5wCkbvabNw']:
  result=subprocess.run(['/Users/bwt/ac-worktrial/.venv/bin/modal','app','stop',app_id,'--yes','--profile','ac-worktrial','--env','worktrial-lance'],capture_output=True,text=True,timeout=30)
  stops.append({'app_id':app_id,'returncode':result.returncode,'stdout':result.stdout[-2000:],'stderr':result.stderr[-2000:]})
  if result.returncode:write('stop_failed_requires_diagnosis',completed_control=current,stops=stops);raise SystemExit(1)
 write('held_after_complete_control',completed_control=current,stops=stops,artifacts='Volumes retained; no code, datasets, checkpoints or results deleted. Any next-source partial attempt remains forensic only; use fresh run IDs after diagnosis.')
 break
