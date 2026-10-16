"""Host driver for six source trials, only after inspected full-repeat proof.

The user keeps this host on. GPU work runs on the already deployed warm class;
no new GPU function or deployment is created by this driver.
"""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BATCH='ab1-math-accuracy-seed20261014'
CONDITIONS=['ruletaker','aegis2','civilcomments','snli-multinli','massive','banking77']

def main():
 import modal
 from modal.config import _set_profile
 _set_profile('ac-worktrial')
 parser=argparse.ArgumentParser();parser.add_argument('--confirmation-driver',default='math-source-repro-driver-v1');args=parser.parse_args()
 volume=modal.Volume.from_name('ac-worktrial-training-results',environment_name='worktrial-lance')
 def read(path):return json.loads(b''.join(volume.read_file(path)))
 gate=read(args.confirmation_driver+'/repeat-gate.json')
 if gate['status']!='passed' or not all(gate['checks'].values()):raise RuntimeError('Exact full-repeat confirmation required')
 baseline=read(args.confirmation_driver+'/control-1.json');commit=baseline['request']['commit'];expected=baseline['worker']['worker_id']
 worker=modal.Cls.from_name('ac-worktrial-math-warm-research','MathWorker',environment_name='worktrial-lance')()
 live=worker.ping.remote()
 if live['failed'] or live['worker_id']!=expected:raise RuntimeError('Original confirmed worker must still be healthy/resident')
 output=ROOT/'outputs/autoresearch/math-source-grid-status.json'
 completed=[]
 def state(phase,**kwargs):
  output.write_text(json.dumps(dict(phase=phase,confirmation_driver=args.confirmation_driver,worker_id=expected,commit=commit,completed=completed,**kwargs),indent=2)+'\n')
 try:
  for name in CONDITIONS:
   state('running',condition=name)
   request={'condition':name,'run_id':BATCH+'-'+name,'commit':commit,'worker_id':expected}
   callfile=ROOT/'outputs/autoresearch'/('math-source-'+name+'-call.json')
   if callfile.exists():
    saved=json.loads(callfile.read_text())
    if saved['request']!=request:raise ValueError('Existing source call identity differs')
    call=modal.FunctionCall.from_id(saved['function_call_id'])
   else:
    call=worker.run.spawn(name,BATCH+'-'+name,commit)
    callfile.write_text(json.dumps({'request':request,'function_call_id':call.object_id},indent=2)+'\n')
   value=call.get()
   if value['status']!='complete' or value['worker']['worker_id']!=expected:raise RuntimeError('Incomplete condition or changed physical worker')
   completed.append(value);state('condition_complete',condition=name)
   (ROOT/'outputs/autoresearch'/('math-source-'+name+'-result.json')).write_text(json.dumps(value,indent=2)+'\n')
   print(json.dumps(value),flush=True)
  state('complete')
 except Exception as exc:state('failed',error=type(exc).__name__,message=str(exc));raise

if __name__=='__main__':main()
