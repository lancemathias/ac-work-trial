"""Persist a Modal call handle before waiting; reuse it on host recovery."""
import json
from pathlib import Path

def invoke(path,request,spawn,restore):
 path=Path(path)
 if path.exists():
  saved=json.loads(path.read_text())
  if saved['request']!=request:raise ValueError('Saved call request differs; preserve it for diagnosis')
  call=restore(saved['function_call_id'])
 else:
  call=spawn()
  # A process failure between spawn and this write needs manual reconciliation;
  # the helper does not claim transactional exactly-once submission.
  with path.open('x') as stream:json.dump({'request':request,'function_call_id':call.object_id},stream,indent=2);stream.write('\n')
 return call.get()
