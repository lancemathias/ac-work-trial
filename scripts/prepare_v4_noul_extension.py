"""Extend v4 with native Boolean training decisions, preserving held-out bytes."""
import collections
import copy
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from jsonschema import Draft202012Validator
from prepare_v4_addbacks import ROOT,canonical,components,normalize,ProtectedTexts,write
from prepare_source_addbacks import words,key
from training_data import digest,load_splits


def condaqa(row,source):
    # Filter before sampling; never turn uncertainty into false.
    if row['label'] not in {'YES','NO'}:return None
    return {'id':'CondaQA:train:'+str(row['SampleID']),'suite':'CondaQA','state':row['sentence1'],
      'question':{'type':'noul','instructions':row['sentence2']},'label':row['label']=='YES',
      'group':'CondaQA:passage:'+str(row['PassageID']),'evaluation_only':False,
      'source':dict(source,source_record_id=str(row['SampleID']),source_group_id=str(row['PassageID']),source_label=row['label'],question_id=row['QuestionID'],passage_edit_id=row['PassageEditID'],raw_record_sha256=hashlib.sha256(canonical(row).encode()).hexdigest(),label_mapping='YES=true; NO=false; all other labels excluded before sampling')}


def creak(row,source):
    if row['label'] not in {'true','false'} or not row['ex_id'].startswith('train_'):raise ValueError('Unexpected CREAK training label/split')
    return {'id':'CREAK:'+row['ex_id'],'suite':'CREAK','state':row['sentence'],
      'question':{'type':'noul','instructions':'Is the claim true?'},'label':row['label']=='true',
      'group':'CREAK:entity:'+str(row['en_wiki_pageid']),'evaluation_only':False,
      'source':dict(source,source_record_id=row['ex_id'],entity_page_id=str(row['en_wiki_pageid']),raw_record_sha256=hashlib.sha256(canonical(row).encode()).hexdigest(),normalization='Native claim and hard Boolean label; annotation explanation excluded from model input')}


def multirc(row,index,source):
    if row['label'] not in [0,1]:raise ValueError('Unexpected MultiRC label')
    return {'id':'MultiRC:'+json.dumps(row['idx'],sort_keys=True),'suite':'MultiRC',
      'state':{'passage':row['paragraph'],'candidate_answer':row['answer']},
      'question':{'type':'noul','instructions':row['question']+' Is the candidate answer correct?'},
      'label':bool(row['label']),'group':'MultiRC:'+str(row['idx']['paragraph']),'evaluation_only':False,
      'source':dict(source,source_row_index=index,native_idx=row['idx'],normalization='Same native answer-correctness representation as frozen parent; no repeated records')}


def prepare():
    spec=json.loads((ROOT/'configs/dataset-v4-expanded-noul.json').read_text());parent=ROOT/spec['parent'];out=ROOT/spec['destination'];receipts=ROOT/spec['receipts']
    if out.exists():raise FileExistsError('Use a fresh destination; releases are immutable')
    if digest(parent)!=spec['parent_manifest_sha256']:raise ValueError('Parent changed')
    _,pm=load_splits(parent)
    native={role:[json.loads(l) for l in (parent.parent/pm['splits'][role]['points']['path']).read_text().splitlines()] for role in ['train','dev','calibration']}
    old=[r for rows in native.values() for r in rows];old_ids={r['id'] for r in old};held_groups={r['group'] for role in ['dev','calibration'] for r in native[role]}
    old_requests={canonical([r['state'],r['question']]) for r in old}
    evidence_groups={r['group'] for r in old if r['suite']=='EvidenceSufficiency'}
    validator=Draft202012Validator(json.loads((ROOT/'configs/normalized-training.schema.json').read_text()))
    pools=collections.defaultdict(list);extras={};sources=[];excluded=collections.Counter()
    for suite,filename,convert,license in [('CondaQA','condaqa_train.json',condaqa,'Apache-2.0'),('CREAK','train.json',creak,'MIT')]:
        folder=ROOT/'data/training/raw/v4-expansion'/suite;source=json.loads((folder/'source.json').read_text())
        for entry in source['files']:
            if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Pinned GitHub source changed')
        sources.append(source)
        raw=(folder/filename).read_text()
        try:rows=json.loads(raw)
        except json.JSONDecodeError:rows=[json.loads(l) for l in raw.splitlines()]
        meta={'repo':source['repo'],'revision':source['revision'],'source_split':'train','license':license,'source_file':filename,'source_file_sha256':digest(folder/filename)}
        for r in rows:
            normalized=convert(r,meta)
            if normalized is None:
                excluded['CondaQA:excluded_label:'+r['label']]+=1;continue
            pools[(suite,'all')].append(normalized)
            if suite=='CondaQA':extras[normalized['id']]=r['original passage']
    raw=ROOT/'work/autoresearch/data/training/raw/MultiRC/multirc/train-00000-of-00001.parquet'
    expected='ced6d731a335a0e4fbd29f6f382f291518952c0f449448361d294000c39e6728'
    if digest(raw)!=expected:raise ValueError('Pinned MultiRC source changed')
    meta={'repo':'aps/super_glue','revision':'3de24cf8022e94f4ee4b9d55a6f539891524d646','source_split':'train','source_file':str(raw.relative_to(ROOT)),'source_file_sha256':expected}
    sources.append(meta)
    for i,r in enumerate(pq.read_table(raw).to_pylist()):
        normalized=multirc(r,i,meta);pools[('MultiRC',str(normalized['label']).lower())].append(normalized)
    prior_inputs=json.loads((parent.parent/'source-inputs.json').read_text())
    shard_root=ROOT/'work/deliverable-audit/untracked-data/cacc/outputs/tasksource-source-audit/filtered-parquet'
    for entry in prior_inputs['filtered_shards']:
        p=ROOT/entry['path']
        if digest(p)!=entry['sha256']:raise ValueError('Pinned filtered shard changed')
        offset=0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=8192):
            for r in batch.to_pylist():
                if r['source']=='procedural-typed-decisions/evidence_sufficiency':
                    result=normalize(r,p.name,entry['sha256'],offset)
                    if result:pools[('EvidenceSufficiency',result[1])].append(result[0])
                offset+=1
    sources.append({'filtered_shards':prior_inputs['filtered_shards']})
    print(canonical({'event':'candidate_pools','counts':{str(k):len(v) for k,v in pools.items()},'excluded_condaqa_labels':dict(excluded)}),flush=True)
    protected=ProtectedTexts();short_exact=set();bench=[]
    for suite,entry in json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites'].items():
        p=ROOT/'data/processed'/(suite+'.jsonl')
        if digest(p)!=entry['sha256']:raise ValueError('Benchmark bytes changed')
        for line in p.open():
            for text in components(json.loads(line)['state']):
                protected.add(text,'benchmark:'+suite)
                if 3<=len(words(text))<8:short_exact.add(key(text))
        bench.append({'suite':suite,'sha256':digest(p)})
    for role in ['dev','calibration']:
        for r in native[role]:
            for text in components(r['state']):
                protected.add(text,'parent:'+role)
                if 3<=len(words(text))<8:short_exact.add(key(text))
    quotas={('CondaQA','all'):2000,('CREAK','all'):1000}
    for label,count in collections.Counter(str(r['label']).lower() for r in native['train'] if r['suite']=='MultiRC').items():quotas[('MultiRC',label)]=count
    for predicate in ['claim_supported','has_conflict']:
        for label in ['false','true']:quotas[('EvidenceSufficiency',predicate+':'+label)]=250
    chosen=[];ids=set(old_ids);requests=set(old_requests);rejections=[];screen_cache={}
    for (suite,stratum),quota in quotas.items():
        accepted=[]
        rank=lambda r:hashlib.sha256(canonical([spec['seed'],suite,stratum,r['id']]).encode()).hexdigest()
        for row in sorted(pools[(suite,stratum)],key=rank):
            reason=None
            request=canonical([row['state'],row['question']])
            if row['id'] in ids:reason='existing or selected ID'
            elif request in requests:reason='duplicate request'
            elif row['group'] in held_groups:reason='parent held-out group'
            elif suite=='EvidenceSufficiency' and row['group'] in evidence_groups:reason='existing or selected procedural evidence group'
            else:
                screen_state={'state':row['state'],'question':row['question']}
                if suite=='CondaQA':screen_state['state']=[row['state'],extras[row['id']]]
                cache_key=canonical(screen_state)
                if cache_key not in screen_cache:
                    screen_cache[cache_key]=protected.hit(screen_state)
                    if suite=='CREAK' and key(row['state']) in short_exact:screen_cache[cache_key]={'kind':'short_claim_exact'}
                reason=screen_cache[cache_key]
            if reason:
                rejections.append({'id':row['id'],'suite':suite,'reason':reason});continue
            validator.validate(row);accepted.append(row);ids.add(row['id']);requests.add(request)
            if suite=='EvidenceSufficiency':evidence_groups.add(row['group'])
            if len(accepted)==quota:break
        if len(accepted)!=quota:raise ValueError(f'Insufficient clean {suite}/{stratum}: {len(accepted)}/{quota}')
        chosen.extend(accepted)
        print(canonical({'event':'selected','suite':suite,'stratum':stratum,'count':len(accepted)}),flush=True)
    assert len(chosen)==4769 and all(r['question']['type']=='noul' for r in chosen)
    out.mkdir(parents=True)
    for role in native:
        payload=(parent.parent/pm['splits'][role]['points']['path']).read_bytes()
        (out/(role+'.jsonl')).write_bytes(payload+(''.join(canonical(r)+'\n' for r in chosen).encode() if role=='train' else b''))
    all_rows=native['train']+chosen+native['dev']+native['calibration']
    (out/'dataset.jsonl').write_text(''.join(canonical(r)+'\n' for r in all_rows))
    (out/'additions.jsonl').write_text(''.join(canonical(r)+'\n' for r in chosen))
    hygiene={'approved':True,'scope':'4769 new training rows; parent hygiene inherited, not independently recertified','dataset_sha256':digest(out/'dataset.jsonl'),'parent_manifest_sha256':digest(parent),'benchmark_pools':bench,'screen':'All eleven full local benchmark pools and parent held-out text components; normalized exact and approximate five-word overlap/containment as previous v4; original CondaQA passage also screened; short CREAK claim exact matching at 3+ words','held_out_groups_excluded':True,'limitations':'Lexical screen cannot exclude unknown paraphrases or certify upstream independence. Existing MultiRC train families may supply new candidate-answer decisions; no held-out families reused.','rejections':rejections,'excluded_condaqa_labels_before_sampling':dict(excluded)}
    write(out/'hygiene-report.json',hygiene)
    manifest=copy.deepcopy(pm);manifest['source_dataset_sha256']=digest(out/'dataset.jsonl');manifest['split_rule']='Exact previous v4 plus4769 screened Noul training rows; dev/calibration byte-identical'
    manifest['hygiene']={'approved':True,'report':{'path':'hygiene-report.json','sha256':digest(out/'hygiene-report.json')}}
    for role in native:
        manifest['splits'][role]['points']={'path':role+'.jsonl','sha256':digest(out/(role+'.jsonl'))};manifest['splits'][role]['count']=len(native[role])+(len(chosen) if role=='train' else 0)
        if role=='train':manifest['splits'][role]['allowed_sources']=sorted(set(manifest['splits'][role]['allowed_sources'])|{'CondaQA','CREAK'})
    write(out/'manifest.json',manifest)
    loaded,_=load_splits(out/'manifest.json');assert {k:len(v) for k,v in loaded.items()}=={'train':14031,'dev':809,'calibration':857}
    for role in ['dev','calibration']:assert (out/(role+'.jsonl')).read_bytes()==(parent.parent/pm['splits'][role]['points']['path']).read_bytes()
    plan=[json.loads(l) for l in (parent.parent/'feature-view-plan.jsonl').read_text().splitlines()]
    plan=[dict(v,cache='reuse_previous_v4') for v in plan]
    plan.extend({'view_id':r['id'],'base_id':r['id'],'group_id':r['group'],'view':'native','source':r['suite'],'type':'noul','cache':'requires_feature_extraction'} for r in chosen)
    assert len(plan)==21524 and len({v['view_id'] for v in plan})==21524
    (out/'feature-view-plan.jsonl').write_text(''.join(canonical(v)+'\n' for v in plan))
    write(out/'source-inputs.json',{'sources':sources})
    def balance(counts):
        total=sum(counts.values());return {'counts':dict(counts),'total':total,'percent':{k:100*v/total for k,v in counts.items()}}
    from run_baselines import load_panel
    type_balance={'native_train':balance(collections.Counter(r['question']['type'] for r in native['train']+chosen)),'training_views':balance(collections.Counter(v['type'] for v in plan)),'benchmark_panel':balance(collections.Counter(r['question']['type'] for r in load_panel())),'counting':'Each example/view/question once; benchmark panel pooled without suite weights'}
    write(out/'type-balance.json',type_balance)
    release={'status':'prepared_text_only','release_id':spec['release_id'],'config':spec,'builder_sha256':digest(__file__),'parent_manifest_sha256':digest(parent),'manifest_sha256':digest(out/'manifest.json'),'dataset_sha256':digest(out/'dataset.jsonl'),'native_counts':{k:len(v) for k,v in loaded.items()},'added_counts':dict(collections.Counter(r['suite'] for r in chosen)),'final_noul_source_counts':dict(collections.Counter(r['suite'] for r in native['train']+chosen if r['question']['type']=='noul')),'planned_training_views':21524,'reused_parent_views':16755,'new_feature_views':4769,'choice_clones_added':0,'unchanged_dev_calibration':True,'actual_loader':'training_data.load_splits passed','model_training_launched':False,'type_balance':type_balance}
    assert release['final_noul_source_counts']==spec['final_noul_source_counts']
    write(out/'release.json',release)
    write(out/'checksums.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    receipts.mkdir(exist_ok=True,parents=True)
    for name in ['release.json','hygiene-report.json','checksums.json','source-inputs.json','type-balance.json']:(receipts/name).write_bytes((out/name).read_bytes())
    (receipts/'selected-additions.jsonl').write_text(''.join(canonical({k:r[k] for k in ['id','suite','label','group','source']})+'\n' for r in chosen))
    print(json.dumps(release,indent=2),flush=True)
    return release

if __name__=='__main__':prepare()
