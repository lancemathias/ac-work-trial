"""Fixed original splits/features, two new training seeds, paired conditions."""
import copy,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEEDS=[20261015,20261016];NAMES=['existing-source-control','massive']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 out=ROOT/'configs/source-confirmation';out.mkdir(exist_ok=True);runs=[]
 for seed in SEEDS:
  for name in NAMES:
   original=ROOT/'configs/source-addbacks'/(name+'.json');cfg=json.loads(original.read_text());manifest=ROOT/'data/training/source-addbacks-v2'/name/'manifest.json';m=json.loads(manifest.read_text())
   if m['split_seed']!=20261014:raise ValueError('Original split seed changed')
   run_id='ab1-math-confirm-seed'+str(seed)+'-'+name
   cfg.update(seed=seed,split_seed=20261014,head_sdpa_backend='math',run_name=run_id,purpose='Preregistered MASSIVE matched new-seed confirmation: training RNG changes only; retain original source records, splits, cached extraction seed20261014,510updates and devaccuracy/earliest-tie selection plus independent calibrationT.')
   target=out/(str(seed)+'-'+name+'.json');target.write_text(json.dumps(cfg,indent=2)+'\n')
   runs.append({'condition':name,'training_seed':seed,'split_seed':20261014,'extraction_seed':20261014,'run_id':run_id,'config':str(target.relative_to(ROOT)),'config_sha256':digest(target),'original_feature_source_config':str(original.relative_to(ROOT)),'original_feature_source_config_sha256':digest(original),'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':digest(manifest),'cache_directory':'/data/hidden-cache/ab1-fast-accuracy-seed20261014-'+name})
 return {'status':'prepared_not_GPU_queued','runs':runs,'success':'Mean paired composite delta>=1pp across two new seeds and positive delta in each; meanCirca delta>=5pp for mechanism support. Original screen seed excluded from confirmation mean.','qualifications':'Fixed repeatedly observed panel; two seeds are limited confirmation, not independent benchmark generalization. Cached source-config identity stays original, separate from new training config.'}
if __name__=='__main__':
 value=prepare();(ROOT/'outputs/autoresearch/massive-confirmation-preparation.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':value['status'],'runs':len(value['runs'])}))
