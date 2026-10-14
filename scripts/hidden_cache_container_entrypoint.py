import json,pathlib,sys
import modal
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from cache_hidden_runtime import extract
snapshot=json.loads((HERE/'source-run.json').read_text());cfg=snapshot['config']['config']
volume=modal.Volume.from_name('ac-worktrial-training-data',environment_name='worktrial-lance')
out=pathlib.Path('/data/hidden-cache/nulun3tv-qwen35-9b-3b3d8db82da7')
try:
 manifest=extract(snapshot,pathlib.Path('/data')/cfg['dataset']['path'].lstrip('/'),HERE/'training-point.schema.json',out,checkpoint=volume.commit)
 print(json.dumps({'status':manifest['status'],'records':manifest['identity']['record_count'],'bytes':manifest['total_bytes']}),flush=True)
finally:volume.commit()
