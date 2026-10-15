"""Prepare the exact250 changed requests for a future fresh feature extraction."""
import copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prepare():
 folder=ROOT/'data/training/ruletaker-cwa-explicit';proof=json.loads((ROOT/'outputs/autoresearch/ruletaker-cwa-cpu.json').read_text())
 cfgpath=ROOT/'configs/source-quality/ruletaker-cwa-explicit.json';cfg=json.loads(cfgpath.read_text())
 if proof['status']!='passed' or sha(cfgpath)!=proof['source_sha256']['configs/source-quality/ruletaker-cwa-explicit.json']:raise ValueError('CPU-tested training config changed')
 rows=[json.loads(x) for x in (folder/'additions.jsonl').read_text().splitlines()]
 old=[json.loads(x) for x in (ROOT/'data/training/source-addbacks-v2/ruletaker/additions.jsonl').read_text().splitlines()]
 if len(rows)!=250 or len(old)!=250 or [x['id'] for x in rows]!=[x['id'] for x in old]:raise ValueError('Expected same250 original IDs/order')
 for a,b in zip(old,rows):
  left,right=copy.deepcopy(a),copy.deepcopy(b)
  old_instruction=left['question'].pop('instructions');new_instruction=right['question'].pop('instructions')
  if old_instruction.split('Hypothesis: ',1)[1]!=new_instruction.split('Hypothesis: ',1)[1]:raise ValueError('Hypothesis changed')
  if left!=right:raise ValueError('More than instruction prefix changed')
 snapshot=json.loads((ROOT/'configs/source-addbacks/union-cache-source.json').read_text())
 snapshot.update(id='ruletaker-cwa250-feature-setup',name='Explicit closed-world RuleTaker250 feature extraction',url=None)
 extract_cfg=copy.deepcopy(cfg);extract_cfg['dataset'].update(path='/project/ruletaker-cwa-explicit/additions.jsonl',sha256=sha(folder/'additions.jsonl'),record_count=250,version='same250-explicit-closed-world-feature-only')
 extract_cfg['purpose']='Feature extraction only for250 changed requests. No training/checkpoint selection; later compose with unchanged v3base and train original accuracy-selected510-update recipe.'
 snapshot['config']['config']=extract_cfg
 output=ROOT/'configs/source-quality/ruletaker-cwa-cache-source.json';output.write_text(json.dumps(snapshot,indent=2)+'\n')
 receipt={'status':'prepared_not_extracted','records':250,'same_ids_labels_and_noninstruction_content':True,'additions_sha256':sha(folder/'additions.jsonl'),'source_snapshot_sha256':sha(output),'training_config_sha256':sha(cfgpath),'base_cache_to_reuse':'/data/hidden-cache/qwen35-9b-v3-fast-v1','fresh_additions_cache':'/data/hidden-cache/ruletaker-cwa250-fast-v1','composition':'Base7928 plus changed250; never reuse originalRuleTaker or union features for changed requests. Compose against full CWA cfg, not extraction-only cfg.','GPU_launch_state':'not queued'}
 return receipt
if __name__=='__main__':
 value=prepare();out=ROOT/'outputs/autoresearch/ruletaker-cwa-cache-preparation.json';out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
