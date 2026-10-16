"""Append screened Boolean views to v4 using the same pinned 27B backbone."""
import copy
import json
import time
from pathlib import Path
from training_data import load_splits, digest
from head_probe_io import write_json, seal

DATASET='readout-feature-dataset-v4-expanded-noul'
PARENT='readout-feature-dataset-v4-wice-evidence-massive'
PROJECT=Path(__file__).resolve().parents[1]
TEXT=PROJECT/'data/training/v4-expanded-noul'


def prepare():
    from resident_backbone import get_processor
    from residual_code_readout import encode,verify_codes
    cfg=json.loads((PROJECT/'configs/readout-mlp/experiment.json').read_text())
    rows,_=load_splits(TEXT/'manifest.json')
    plan=[json.loads(line) for line in (TEXT/'feature-view-plan.jsonl').read_text().splitlines()]
    by_id={r['id']:r for r in rows['train']}
    tokenizer=get_processor(cfg).tokenizer
    verify_codes(tokenizer,cfg['backbone_id'])
    prepared=[]
    for view in plan:
        if view['cache']!='requires_feature_extraction':continue
        row=by_id[view['base_id']]
        item=encode(tokenizer,row,True,'native' if view['view']=='native' else 'shuffle',cfg['seed'],0 if view['view']=='native' else 1)
        item.update(id=view['view_id'],base_id=view['base_id'],group_id=view['group_id'],view=view['view'])
        prepared.append(item)
    assert len(prepared)==4769 and all(r['type']=='noul' and r['view']=='native' for r in prepared)
    return cfg,prepared,{'status':'passed','backbone_id':cfg['backbone_id'],'backbone_revision':cfg['backbone_revision'],'text_manifest_sha256':digest(TEXT/'manifest.json'),'new_views':len(prepared),'max_tokens':max(len(r['input_ids']) for r in prepared),'tokens':sum(len(r['input_ids']) for r in prepared),'no_truncation':True,'batch_size':4,'shuffle_seed':cfg['seed'],'shuffle_epoch':1}


def run(results,commit):
    import torch
    from resident_backbone import get_model
    from kernel_runtime import configure_optimized_runtime
    from train_residual_readout import extract
    from finish_fixed_readout import verified_cache
    from cached_v31 import save_cache
    root=Path(results);out=root/DATASET
    cfg,prepared,proof=prepare()
    parent=root/PARENT
    pm=json.loads((parent/'manifest.json').read_text())
    if (pm['backbone_id'],pm['backbone_revision'])!=(cfg['backbone_id'],cfg['backbone_revision']):raise ValueError('Backbone differs from v3.1')
    if pm['codes_sha256']!=digest(PROJECT/'configs/readout-mlp/codes-255.json'):raise ValueError('Answer codes differ')
    if (out/'result.json').exists():
        verified_cache(out/'train')
        return json.loads((out/'result.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'preflight.json',proof)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
    write_json(out/'kernel-runtime.json',configure_optimized_runtime('both'))
    backbone=get_model(cfg);versions=[(p,p._version) for p in backbone.parameters()]
    additions=extract(out/'additions',prepared,backbone,getattr(__import__('resident_backbone',fromlist=['get_processor']),'get_processor')(cfg).tokenizer,commit,time.time()+11*3600)
    if any(p.requires_grad or p.grad is not None or p._version!=v for p,v in versions):raise RuntimeError('Frozen backbone changed')
    native=verified_cache(parent/'train')
    expected=[json.loads(x) for x in (TEXT/'feature-view-plan.jsonl').read_text().splitlines()]
    items=native[1]+additions[1]
    if len(items)!=21524 or {r['id'] for r in items}!={r['view_id'] for r in expected}:raise ValueError('v4 view inventory mismatch')
    references={}
    request=dict(proof,parent_manifest_sha256=digest(parent/'manifest.json'))
    save_cache(out/'train',(torch.cat([native[0],additions[0]]),items),request)
    for role in ['dev','calibration']:
        old=verified_cache(parent/role)
        save_cache(out/role,old,dict(request,role=role,unchanged=True))
        references[role]={'parent_checksums_sha256':digest(parent/role/'checksums.json'),'records':len(old[1]),'identical_tensors_and_items':True}
    manifest={'version':'v4','format':pm['format'],'backbone_id':cfg['backbone_id'],'backbone_revision':cfg['backbone_revision'],'codes_sha256':pm['codes_sha256'],'parent':PARENT,'parent_manifest_sha256':digest(parent/'manifest.json'),'text_manifest_sha256':proof['text_manifest_sha256'],'train_views':21524,'new_views':4769,'reused_views':16755,'references':references,'test_reference':pm['test_reference'],'no_model_training':True,'source_sha256':{n:digest(PROJECT/'scripts'/n) for n in ['cache_v4_extension.py','train_residual_readout.py','residual_code_readout.py','resident_backbone.py']}}
    write_json(out/'manifest.json',manifest)
    result={'status':'complete',**manifest}
    write_json(out/'result.json',result);seal(out);commit()
    import tarfile
    exports=root/'exports';exports.mkdir(exist_ok=True)
    archive=exports/(DATASET+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=DATASET)
    receipt={'archive_path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),'result':result}
    write_json(exports/(DATASET+'-export.json'),receipt);commit()
    return receipt
