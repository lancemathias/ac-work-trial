"""Fetch completion and export receipt for the v4 Modal call."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'outputs/data-releases/v4-wice-evidence-massive'
launch=json.loads((folder/'cache-launch.json').read_text())
call=modal.FunctionCall.from_id(launch['function_call_id'])
try:
    result=call.get(timeout=0)
except TimeoutError:
    print(json.dumps({'status':'running','function_call_id':call.object_id}))
else:
    (folder/'cache-completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
