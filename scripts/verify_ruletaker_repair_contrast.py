"""Explicit fresh-parent-cache exception for the same-record instruction repair."""
import hashlib,json
from pathlib import Path
from verify_source_training_contrast import verify as source_verify

def read_json(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parent_checks(original,candidate,base,fresh,release):
 a,b=original['feature_cache']['identity'],candidate['feature_cache']['identity'];pa,pb=a['parent_manifest_sha256'],b['parent_manifest_sha256']
 # Fixed frozen numerical identity, explicitly enumerated as in the source screen.
 keys=['backbone_id','backbone_revision','encoder_sha256','adapter_sha256','dtype','hidden_layer','attention_implementation','extraction_batch_size','padding','max_input_tokens','no_truncation','torch','transformers','hidden_size','head_config','tf32','cudnn_sdp','attention_mode','attention_hook_sha256','optimized_kernel_runtime']
 checks={'frozen_extraction_regime_exact':all(a[k]==b[k] for k in keys),'same_base_parent':len(pa)==len(pb)==2 and pa[0]==pb[0],
 'fresh_addition_parent':len(pa)==len(pb)==2 and pa[1]!=pb[1],
 'both_parents_complete_and_backward_verified':all(x.get('status')=='complete' and x.get('cached_head_backward_verified') is True for x in [base,fresh]),
 'base7928_records':base['identity']['record_count']==7928,'fresh250_records':fresh['identity']['record_count']==250,
 'fresh_request_bytes_match_preparation':fresh['identity']['dataset_sha256']==release['additions_sha256'],
 'full_training_dataset_matches_preparation':candidate['config']['dataset']['sha256']==release['full_dataset_sha256']}
 return checks

def verify(original_training,candidate_training,base_manifest,fresh_manifest,preparation,cpu_dataset_proof):
 original_training,candidate_training=Path(original_training),Path(candidate_training)
 # This performs checksum checks and retains every original non-cache protocol check.
 prior=source_verify(original_training,candidate_training);checks=dict(prior['checks']);del checks['frozen_extraction_regime_and_parents_exact']
 a=read_json(original_training/'config.json');b=read_json(candidate_training/'config.json');parents=b['feature_cache']['identity']['parent_manifest_sha256']
 base,fresh=read_json(base_manifest),read_json(fresh_manifest);proof=read_json(cpu_dataset_proof)
 if proof['status']!='passed':raise ValueError('Passed same-record CPU proof required')
 release=dict(read_json(preparation),full_dataset_sha256=proof['dataset_sha256'])
 checks.update(parent_checks(a,b,base,fresh,release))
 pinned=read_json(Path(__file__).resolve().parents[1]/'configs/source-addbacks/ruletaker.json')
 checks['original_is_pinned_ruletaker_dataset']=a['config']['dataset']['sha256']==pinned['dataset']['sha256']
 checks['actual_base_manifest_sha_matches_training']=digest(base_manifest)==parents[0]
 checks['actual_fresh_manifest_sha_matches_training']=digest(fresh_manifest)==parents[1]
 for root,key in [(original_training,'a'),(candidate_training,'b')]:
  raw=(root/'metrics.jsonl').read_bytes()
  if hashlib.sha256(raw).hexdigest()!=read_json(root/'checksums.json')['metrics.jsonl']:raise ValueError('Training trace changed')
  values=[json.loads(x) for x in raw.decode().splitlines()];windows=[x['record_ids'] for x in values if x['event']=='train_step']
  if key=='a':original_windows=windows
  else:checks['all510_training_record_windows_exact']=windows==original_windows
 return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,'scope':'Prospective same-record RuleTaker instruction repair only: fresh250-request parent explicitly replaces original union parent; same7928 base and frozen numerical regime. Original source-screen verifier is unchanged. No performance or cross-seed conclusion.'}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser()
 for name in ['original_training','candidate_training','base_manifest','fresh_manifest','preparation','cpu_dataset_proof','output']:p.add_argument(name)
 a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Preserve prior verification')
 value=verify(a.original_training,a.candidate_training,a.base_manifest,a.fresh_manifest,a.preparation,a.cpu_dataset_proof);out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
 if value['status']!='passed':raise RuntimeError('Repair contrast did not match the preregistered protocol')
