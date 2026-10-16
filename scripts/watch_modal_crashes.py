"""Bounded local crash-loop circuit breaker for this research's Modal apps."""
import concurrent.futures,datetime,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODAL=Path('/Users/bwt/ac-worktrial/.venv/bin/modal')
OUT=ROOT/'outputs/autoresearch';END=1792125869
FLAGS=['--profile','ac-worktrial','--env','worktrial-lance']
def emit(event,**kwargs):
 row={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kwargs}
 with (OUT/'crash-watch.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def registry():
 ids={'ap-2QEsgQ3Ord7fXG7ugKjJzi'}
 for p in OUT.glob('*-launch.json'):
  row=json.loads(p.read_text())
  if row.get('modal_app'):ids.add(row['modal_app'])
 return ids
def logs(app_id):
 try:
  p=subprocess.run([str(MODAL),'app','logs',app_id,*FLAGS],capture_output=True,text=True,timeout=5)
  text=p.stdout+'\n'+p.stderr
 except subprocess.TimeoutExpired as exc:
  def decode(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
  text=decode(exc.stdout)+'\n'+decode(exc.stderr)
 # One startup failure is enough: Modal may restart these containers automatically.
 reason='container_startup_failure' if 'Runner failed with exception' in text else 'repeated_unhandled_exception' if text.count('Traceback (most recent call last)')>=2 else None
 return app_id,reason,text
emit('watchdog_started',interval_seconds=20,scope='Research app IDs only; stop on first startup failure or repeated traceback')
last_errors=set();last_heartbeat=0
while time.time()<END:
 try:
  p=subprocess.run([str(MODAL),'app','list',*FLAGS,'--json'],capture_output=True,text=True,timeout=10,check=True)
  apps=json.loads(p.stdout);active={a['app_id'] for a in apps if a['state']!='stopped'}&registry()
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for app_id,reason,text in pool.map(logs,sorted(active)):
    if reason:
     evidence=OUT/(app_id+'-failure.log');evidence.write_text(text[-30000:])
     stopped=subprocess.run([str(MODAL),'app','stop',app_id,'--yes',*FLAGS],capture_output=True,text=True,timeout=15)
     emit('circuit_breaker_stop',app_id=app_id,reason=reason,returncode=stopped.returncode,evidence=evidence.name,detail=stopped.stderr[-1500:])
  if time.time()-last_heartbeat>300:
   (OUT/'crash-watch-status.json').write_text(json.dumps({'checked_at':time.time(),'active_apps':sorted(active),'state':'monitoring'},indent=2)+'\n');last_heartbeat=time.time()
 except Exception as exc:
  key=(type(exc).__name__,str(exc))
  if key not in last_errors:emit('watchdog_inspection_error',error=key[0],detail=key[1][:1500]);last_errors.add(key)
 time.sleep(20)
emit('watchdog_reservation_window_ended')
