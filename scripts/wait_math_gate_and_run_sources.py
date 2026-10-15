"""Bounded host handoff: inspect full repeat proof, then run held source trials."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={'initial-head.safetensors','best-head.safetensors','best.json','temperature.json','dev-traces.jsonl','calibration-traces.jsonl','all_training_metrics','510_updates_each','raw-predictions.jsonl','calibrated-predictions.jsonl'}
def main():
 import modal
 from modal.config import _set_profile
 _set_profile('ac-worktrial');v=modal.Volume.from_name('ac-worktrial-training-results',environment_name='worktrial-lance')
 folder='math-source-repro-driver-v1';out=ROOT/'outputs/autoresearch/math-source-handoff-status.json';until=time.time()+9000
 def read(name):return json.loads(b''.join(v.read_file(folder+'/'+name)))
 def status(phase,**kwargs):out.write_text(json.dumps(dict(phase=phase,time=time.time(),**kwargs),indent=2)+'\n')
 try:
  status('waiting_for_full_repeat_gate')
  while time.time()<until:
   try:gate=read('repeat-gate.json')
   except FileNotFoundError:
    state=read('status.json')
    if state['phase']=='failed':raise RuntimeError('Control driver failed: '+str(state))
    time.sleep(20);continue
   if gate['status']!='passed' or set(gate['checks'])!=REQUIRED or not all(x is True for x in gate['checks'].values()):raise RuntimeError('Full exact-repeat proof did not pass')
   first,second=read('control-1.json'),read('control-2.json')
   if first['worker']['worker_id']!=second['worker']['worker_id'] or first['initial_head_sha256']!=second['initial_head_sha256']:raise RuntimeError('Matched worker/initialization proof differs')
   if first['model_loads']!=second['model_loads'] or second['model_loads']!=1:raise RuntimeError('Expected one retained model load')
   proof={'status':'passed','gate':gate,'first':first,'second':second,'inspection':'All10 preregistered full-run checks true; same worker and one model load; first control remains baseline regardless of score.'}
   (ROOT/'outputs/autoresearch/math-source-full-repeat-proof.json').write_text(json.dumps(proof,indent=2)+'\n');status('source_grid_running',proof=proof)
   result=subprocess.run([sys.executable,str(ROOT/'scripts/run_math_source_grid.py')],cwd=ROOT)
   if result.returncode:raise RuntimeError('Source grid stopped; inspect its status, never retry automatically')
   status('complete');return
  raise TimeoutError('Bounded wait expired; inspect original driver, never restart it automatically')
 except Exception as exc:status('failed',error=type(exc).__name__,message=str(exc));raise
if __name__=='__main__':main()
