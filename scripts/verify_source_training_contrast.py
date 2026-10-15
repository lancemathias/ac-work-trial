"""Check actual exported source runs match the preregistered training protocol."""
import hashlib,json
from pathlib import Path

def verify(control,candidate):
 control,candidate=Path(control),Path(candidate)
 def read(root,name):
  hashes=json.loads((root/'checksums.json').read_text());raw=(root/name).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=hashes[name]:raise ValueError('Changed exported evidence: '+name)
  return json.loads(raw)
 ca,cb=read(control,'config.json'),read(candidate,'config.json');sa,sb=read(control,'summary.json'),read(candidate,'summary.json');a,b=ca['config'],cb['config']
 checks={};variable={'dataset','purpose','run_name'}
 checks['held_recipe_exact']={k:v for k,v in a.items() if k not in variable}=={k:v for k,v in b.items() if k not in variable}
 checks['accuracy_selection']=a['selection_metric']==b['selection_metric']=='accuracy'
 checks['head_only_math']=a['head_sdpa_backend']==b['head_sdpa_backend']=='math'
 checks['8178_total_records']=a['dataset']['record_count']==b['dataset']['record_count']==8178
 checks['510_updates']=sa['steps']==sb['steps']==510
 checks['frozen_features_and_backbone']=all(s['backbone_parameter_versions_unchanged'] and s['cached_features_unchanged'] for s in [sa,sb])
 for role in ['dev','calibration']:checks[role+'_bytes_exact']=ca['manifest']['splits'][role]['points']['sha256']==cb['manifest']['splits'][role]['points']['sha256']
 for name in ['torch','transformers','head_source_sha256','training_point_schema_sha256','code_sha256','cache_reader_sha256']:checks[name+'_exact']=ca[name]==cb[name]
 keys=['backbone_id','backbone_revision','encoder_sha256','adapter_sha256','dtype','hidden_layer','attention_implementation','extraction_batch_size','padding','max_input_tokens','no_truncation','torch','transformers','hidden_size','head_config','tf32','cudnn_sdp','attention_mode','attention_hook_sha256','optimized_kernel_runtime','parent_manifest_sha256']
 ia,ib=ca['feature_cache']['identity'],cb['feature_cache']['identity'];checks['frozen_extraction_regime_and_parents_exact']=all(ia[k]==ib[k] for k in keys)
 initial=[];traces=[]
 for root in [control,candidate]:
  hashes=json.loads((root/'checksums.json').read_text());sha=hashlib.sha256((root/'initial-head.safetensors').read_bytes()).hexdigest()
  if sha!=hashes['initial-head.safetensors']:raise ValueError('Changed initial checkpoint')
  initial.append(sha);raw=(root/'metrics.jsonl').read_bytes()
  if hashlib.sha256(raw).hexdigest()!=hashes['metrics.jsonl']:raise ValueError('Changed training trace')
  traces.append([json.loads(x) for x in raw.decode().splitlines() if json.loads(x)['event']=='train_step'])
 checks['initial_head_exact']=initial[0]==initial[1]=='909b3c9ff1f565479d2bd81a1346564af2f5f0a433f2bf869fa9e7cad4ed573e'
 checks['lr_schedule_exact']=[r['learning_rate'] for r in traces[0]]==[r['learning_rate'] for r in traces[1]]
 checks['actual_decisions_each_epoch_6512']=all(len(rows)==510 and all(sum(r['decisions'] for r in rows if r['epoch']==e)==6512 for e in range(1,6)) for rows in traces)
 return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,'control':str(control.resolve()),'candidate':str(candidate.resolve()),'scope':'Dataset/source additions differ as preregistered; all listed recipe/runtime/initialization/heldout/schedule properties fixed. Does not establish source-effect robustness across seeds.'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('control');p.add_argument('candidate');p.add_argument('output');a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError('Preserve previous contrast verification')
 result=verify(a.control,a.candidate);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if result['status']!='passed':raise RuntimeError('Training contrast does not match protocol; do not attribute a source effect')
