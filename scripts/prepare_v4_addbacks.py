"""Prepare v4 text addbacks, preserving WiCE Choice and procedural Noul semantics."""
import collections
import copy
import hashlib
import json
import re
from pathlib import Path
import pyarrow.parquet as pq
from jsonschema import Draft202012Validator
from prepare_source_addbacks import words, texts, key
from training_data import canonical as base_canonical, digest, load_splits

def canonical(value):
    # str.splitlines() in the actual loader treats Unicode separators as record ends.
    return base_canonical(value).replace('\u0085','\\u0085').replace('\u2028','\\u2028').replace('\u2029','\\u2029')

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'winogrande','financialphrasebank','ragtruth','judgebench','bbh','bigbenchhard','jevbench','tabfact','contractnli','circa','belebele','truthfulqa'}
PAIRS=[('text_A','text_B'),('First text','Second text'),('Passage A','Passage B'),('A','B'),('Premise','Hypothesis')]


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def components(state):
    values=list(texts(state))
    extra=[]
    for text in values:
        for left,right in PAIRS:
            match=re.match(r'\A'+re.escape(left)+r':\s*(.*?)\n\s*'+re.escape(right)+r':\s*(.*)\Z',text,re.S)
            if match:
                extra.extend(match.groups())
                break
    return list(dict.fromkeys(values+extra))


class ProtectedTexts:
    def __init__(self):
        self.exact=collections.defaultdict(set)
        self.sets={}
        self.anchors=collections.defaultdict(set)

    @staticmethod
    def grams(tokens):
        return {hashlib.sha256(' '.join(tokens[i:i+5]).encode()).digest()[:12] for i in range(len(tokens)-4)}

    def add(self,text,label):
        tokens=words(text)
        if len(tokens)<8:return
        k=key(text);self.exact[k].add(label)
        if len(tokens)<20 or k in self.sets:return
        grams=self.grams(tokens);self.sets[k]=grams
        for anchor in sorted(grams)[:8]:self.anchors[anchor].add(k)

    def hit(self,row):
        for text in components(row['state'])+list(texts(row['question'])):
            tokens=words(text)
            if len(tokens)<8:continue
            k=key(text)
            if k in self.exact:return {'kind':'normalized_exact','protected':sorted(self.exact[k])}
            if len(tokens)<20:continue
            grams=self.grams(tokens)
            possible={k for g in grams for k in self.anchors.get(g,())}
            for k in sorted(possible):
                other=self.sets[k];shared=len(grams&other)
                if shared>=20 and (shared/len(grams|other)>=.85 or shared/min(len(grams),len(other))>=.95):
                    return {'kind':'near_or_containment','protected':sorted(self.exact[k])}
        return None


def normalize(row,shard,shard_hash,index):
    if row['split']!='train' or row['variant']!='direct':return None
    source=row['source']
    meta={'dataset_repo':'tasksource/tasksource-jev-typed-decisions','dataset_config':'filtered-full',
          'original_pool_revision':'d2ab1d12be4463fb7ac1f877ed2193b3bb59642a',
          'native_source':source,'source_split':'train','source_record_id':row['id'],
          'source_group_id':row['group_id'],'source_question_id':row['question_id'],
          'local_shard':shard,'local_shard_sha256':shard_hash,'source_row_index':index,
          'upstream_row_sha256':hashlib.sha256(canonical(row).encode()).hexdigest(),
          'license':row['license'],'license_use':row['license_use'],'variant':row['variant']}
    if row['license_use']!='commercial':raise ValueError('Non-commercial row in commercial-filtered input')
    if source=='wice':
        if row['kind']!='choice':return None
        if set(row['options'])!={'supported','partially_supported','not_supported'} or len(row['options'])!=3:
            raise ValueError('Unexpected WiCE labels')
        if len(row['target'])!=3 or sum(row['target'])!=1. or any(v not in [0.,1.] for v in row['target']):
            raise ValueError('WiCE requires native hard target')
        label=row['options'][row['target'].index(1.)]
        question={'type':'choice','instructions':row['question'],'criteria':{f'option_{i}':v for i,v in enumerate(row['options'])}}
        gold=f"option_{row['target'].index(1.)}"
        parsed=components(row['state'])
        if len(parsed)<3:raise ValueError('Cannot recover WiCE evidence/claim boundaries')
        group='WiCE:document:'+key(parsed[1])
        suite='WiCE';stratum=label
    elif source=='procedural-typed-decisions/evidence_sufficiency':
        if row['kind']!='noul':return None
        if row['question_id'] not in {'claim_supported','has_conflict'}:raise ValueError('Unexpected Noul predicate')
        if row['options'] or len(row['target'])!=1 or row['target'][0] not in [0.,1.]:raise ValueError('Expected hard native Boolean')
        suite='EvidenceSufficiency';gold=bool(row['target'][0]);stratum=row['question_id']+':'+str(gold).lower()
        question={'type':'noul','instructions':row['question']}
        group='EvidenceSufficiency:'+row['group_id']
    elif source=='multilingual/massive':
        if row['kind']!='choice':return None
        if len(row['options'])!=60 or len(set(row['options']))!=60:raise ValueError('MASSIVE requires 60 intents')
        if len(row['target'])!=60 or sum(row['target'])!=1. or any(v not in [0.,1.] for v in row['target']):raise ValueError('Expected hard intent target')
        suite='MASSIVE';stratum='all';gold=f"option_{row['target'].index(1.)}"
        question={'type':'choice','instructions':row['question'],'criteria':{f'option_{i}':v for i,v in enumerate(row['options'])}}
        group='MASSIVE:filtered:'+row['group_id']
    else:return None
    return {'id':'Tasksource:'+row['id'],'suite':suite,'state':row['state'],'question':question,'label':gold,
            'group':group,'source':meta,'evaluation_only':False},stratum


def prepare(config_path):
    spec=json.loads(Path(config_path).read_text());parent=ROOT/spec['parent'];out=ROOT/spec['destination'];receipts=ROOT/spec['receipts']
    if out.exists():raise FileExistsError('Preserve immutable releases; use a fresh destination')
    if digest(parent)!=spec['parent_manifest_sha256']:raise ValueError('E8/E9 parent changed')
    parent_rows,pm=load_splits(parent)
    native={role:[json.loads(x) for x in (parent.parent/pm['splits'][role]['points']['path']).read_text().splitlines()] for role in ['train','dev','calibration']}
    all_parent=[r for rr in native.values() for r in rr]
    parent_ids={r['id'] for r in all_parent};parent_groups={r['group'] for r in all_parent}
    parent_keys={key(t) for r in all_parent for t in components(r['state']) if len(words(t))>=8}
    validator=Draft202012Validator(json.loads((ROOT/'configs/normalized-training.schema.json').read_text()))
    paths=sorted((ROOT/spec['filtered_parquet']).glob('*.parquet'))
    source_receipts=[];pool=collections.defaultdict(list);source_counts=collections.Counter()
    for path in paths:
        hashed=digest(path);source_receipts.append({'path':str(path.relative_to(ROOT)),'sha256':hashed,'bytes':path.stat().st_size})
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                if row['source'] in {'wice','procedural-typed-decisions/evidence_sufficiency','multilingual/massive'}:
                    source_counts[(row['source'],row['kind'],row['variant'])]+=1
                    converted=normalize(row,path.name,hashed,offset)
                    if converted:pool[(converted[0]['suite'],converted[1])].append(converted[0])
                offset+=1
    print(canonical({'event':'candidate_pools','counts':{str(k):len(v) for k,v in pool.items()}}),flush=True)
    protected=ProtectedTexts();bench_receipts=[]
    for suite,entry in json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites'].items():
        p=ROOT/'data/processed'/(suite+'.jsonl')
        if digest(p)!=entry['sha256']:raise ValueError('Benchmark pool changed')
        count=0
        for line in p.open():
            count+=1;r=json.loads(line)
            for text in components(r['state']):protected.add(text,'benchmark:'+suite)
        bench_receipts.append({'suite':suite,'records':count,'sha256':digest(p)})
    for role in ['dev','calibration']:
        for r in native[role]:
            for text in components(r['state']):protected.add(text,'parent:'+role)
    print(canonical({'event':'protected_index','benchmark_pools':len(bench_receipts),'unique_texts':len(protected.exact)}),flush=True)
    rejections=[];selected=[];selected_ids=set();selected_procedural_groups=set();eligibility={}
    for suite in ['WiCE','EvidenceSufficiency']:
        for stratum,quota in spec['quotas'][suite].items():
            def rank(row):return hashlib.sha256(canonical([spec['seed'],suite,stratum,row['id']]).encode()).hexdigest()
            accepted=[]
            for row in sorted(pool[(suite,stratum)],key=rank):
                reason=None
                if row['id'] in parent_ids or row['group'] in parent_groups:reason='parent ID/group'
                elif row['id'] in selected_ids:reason='duplicate input ID'
                elif suite=='EvidenceSufficiency' and row['group'] in selected_procedural_groups:reason='one procedural question per evidence group'
                elif any(key(t) in parent_keys for t in components(row['state']) if len(words(t))>=8):reason='parent exact context'
                else:
                    contentkey=key(row['state'])
                    if contentkey not in eligibility:eligibility[contentkey]=protected.hit(row)
                    reason=eligibility[contentkey]
                if reason:
                    rejections.append({'id':row['id'],'suite':suite,'stratum':stratum,'reason':reason});continue
                validator.validate(row);accepted.append(row);selected_ids.add(row['id'])
                if suite=='EvidenceSufficiency':selected_procedural_groups.add(row['group'])
                if len(accepted)==quota:break
            if len(accepted)!=quota:raise ValueError(f'Insufficient clean {suite}/{stratum}: {len(accepted)}/{quota}')
            selected.extend(accepted)
            print(canonical({'event':'selected','suite':suite,'stratum':stratum,'count':len(accepted)}),flush=True)
    massive=ROOT/spec['massive_additions'];legacy_manifest=json.loads((massive.parent/'manifest.json').read_text())
    if digest(massive.parent/'dataset.jsonl')!=legacy_manifest['source_dataset_sha256']:raise ValueError('Overnight MASSIVE bytes changed')
    massive_rows=[json.loads(x) for x in massive.read_text().splitlines()]
    if len(massive_rows)!=250:raise ValueError('Expected exact 250 original MASSIVE decisions')
    for row in massive_rows:
        if row['suite']!='MASSIVE' or row['question']['type']!='choice' or len(row['question']['criteria'])!=60:raise ValueError('MASSIVE semantics changed')
        if row['id'] in parent_ids or row['group'] in parent_groups or row['id'] in selected_ids:raise ValueError('MASSIVE parent overlap')
        if any(key(t) in parent_keys for t in components(row['state']) if len(words(t))>=8) or protected.hit(row):raise ValueError('Original MASSIVE now overlaps protected material')
        validator.validate(row)
    original_massive_count=len(massive_rows)
    massive_keys={key(r['state']) for r in massive_rows}
    massive_ids={r['id'] for r in massive_rows}
    candidates=sorted(pool[('MASSIVE','all')],key=lambda r:hashlib.sha256(canonical([spec['seed'],'MASSIVE',r['id']]).encode()).hexdigest())
    for row in candidates:
        if len(massive_rows)==spec['quotas']['MASSIVE']:break
        reason=None
        if row['id'] in parent_ids|selected_ids|massive_ids or row['group'] in parent_groups:reason='duplicate parent or addition ID/group'
        elif key(row['state']) in massive_keys:reason='duplicate MASSIVE utterance'
        elif any(key(t) in parent_keys for t in components(row['state']) if len(words(t))>=8):reason='parent exact context'
        else:reason=protected.hit(row)
        if reason:
            rejections.append({'id':row['id'],'suite':'MASSIVE','reason':reason});continue
        validator.validate(row);massive_rows.append(row);massive_keys.add(key(row['state']));massive_ids.add(row['id'])
    if len(massive_rows)!=spec['quotas']['MASSIVE']:raise ValueError('Insufficient clean MASSIVE examples')
    selected.extend(massive_rows)
    assert len(selected)==3000 and len({r['id'] for r in selected})==3000
    for row in selected:
        meta=re.sub('[^a-z0-9]','',json.dumps(row['source']).casefold())
        if any(s in meta for s in FORBIDDEN):raise ValueError('Excluded source lineage')
    out.mkdir(parents=True)
    for role in native:
        src=parent.parent/pm['splits'][role]['points']['path']
        (out/(role+'.jsonl')).write_bytes(src.read_bytes()+(''.join(canonical(r)+'\n' for r in selected).encode() if role=='train' else b''))
    (out/'additions.jsonl').write_text(''.join(canonical(r)+'\n' for r in selected))
    combined=native['train']+selected+native['dev']+native['calibration']
    (out/'dataset.jsonl').write_text(''.join(canonical(r)+'\n' for r in combined))
    proof={'approved':True,'scope':'New additions only; frozen parent inherits its recorded hygiene. All eleven full local benchmark pools plus parent held-outs screened; no benchmark labels used.',
       'dataset_sha256':digest(out/'dataset.jsonl'),'parent_manifest_sha256':digest(parent),'benchmark_pools':bench_receipts,
       'screen':{'normalized_exact_min_words':8,'shingle_words':5,'min_shared_shingles':20,'jaccard':.85,'containment':.95,'minhash_anchors':8,'paired_text_components':True},
       'limitations':'Lexical matching is approximate and cannot certify unknown paraphrases/upstream derivations. Fixed held-outs contain no new-source examples.',
       'rejections':rejections,'selected_counts':dict(collections.Counter(r['suite'] for r in selected))}
    write(out/'hygiene-report.json',proof)
    manifest=copy.deepcopy(pm);manifest['source_dataset_sha256']=proof['dataset_sha256'];manifest['split_rule']='Exact E8/E9 parent; 3000 screened training-only additions; dev/calibration byte-identical'
    manifest['hygiene']={'approved':True,'report':{'path':'hygiene-report.json','sha256':digest(out/'hygiene-report.json')}}
    for role in native:
        manifest['splits'][role]['points']={'path':role+'.jsonl','sha256':digest(out/(role+'.jsonl'))}
        manifest['splits'][role]['count']=len(native[role])+(len(selected) if role=='train' else 0)
        if role=='train':manifest['splits'][role]['allowed_sources']=sorted(set(manifest['splits'][role]['allowed_sources'])|{'WiCE','EvidenceSufficiency','MASSIVE'})
    write(out/'manifest.json',manifest)
    loaded,_=load_splits(out/'manifest.json')
    assert {k:len(v) for k,v in loaded.items()}=={'train':9262,'dev':809,'calibration':857}
    for role in ['dev','calibration']:assert (out/(role+'.jsonl')).read_bytes()==(parent.parent/pm['splits'][role]['points']['path']).read_bytes()
    # View inventory is a plan, not a frozen-feature cache.
    parent_train_ids={r['id'] for r in native['train']};views=[]
    for row in native['train']+selected:
        views.append({'view_id':row['id'],'base_id':row['id'],'group_id':row['group'],'view':'native','source':row['suite'],'type':row['question']['type'],'cache':'reuse_parent_native' if row['id'] in parent_train_ids else 'requires_feature_extraction'})
        if row['question']['type']=='choice':
            parent_row=row['id'] in parent_train_ids
            views.append({'view_id':row['id']+('::v3.1-shuffle-1' if parent_row else '::v4-shuffle-1'),'base_id':row['id'],'group_id':row['group'],'view':'shuffle-1','source':row['suite'],'type':'choice','seed':spec['seed'],'epoch':1,'cache':'reuse_parent_permutation' if parent_row else 'requires_feature_extraction'})
    assert len(views)==16755 and len({v['view_id'] for v in views})==16755
    (out/'feature-view-plan.jsonl').write_text(''.join(canonical(v)+'\n' for v in views))
    write(out/'source-inputs.json',{'filtered_shards':source_receipts,'massive_additions':{'path':str(massive.relative_to(ROOT)),'sha256':digest(massive)},'source_variant_counts':{str(k):v for k,v in source_counts.items()}})
    release={'status':'prepared_text_only','release_id':spec['release_id'],'config':spec,'config_sha256':digest(config_path),'builder_sha256':digest(__file__),
       'parent_manifest_sha256':digest(parent),'manifest_sha256':digest(out/'manifest.json'),'dataset_sha256':digest(out/'dataset.jsonl'),
       'native_counts':{role:len(rows) for role,rows in loaded.items()},'added_counts':dict(collections.Counter(r['suite'] for r in selected)),
       'added_types':dict(collections.Counter(r['question']['type'] for r in selected)),'train_option_counts':dict(collections.Counter(len(r['question'].get('criteria',{})) if r['question']['type']=='choice' else 2 for r in native['train']+selected)),
       'planned_training_views':len(views),'reused_parent_views':sum(v['cache'].startswith('reuse') for v in views),'new_feature_views':sum(v['cache']=='requires_feature_extraction' for v in views),
       'unchanged_dev_calibration':True,'actual_loader':'training_data.load_splits passed','model_training_launched':False,'benchmark_screen_rejections':len(rejections)}
    write(out/'release.json',release)
    checksums={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    write(out/'checksums.json',checksums)
    receipts.mkdir(parents=True,exist_ok=True)
    for name in ['release.json','hygiene-report.json','source-inputs.json','checksums.json']:(receipts/name).write_bytes((out/name).read_bytes())
    (receipts/'selected-additions.jsonl').write_text(''.join(canonical({'id':r['id'],'source':r['suite'],'type':r['question']['type'],'label':r['label'],'group':r['group'],'provenance':r['source']})+'\n' for r in selected))
    print(json.dumps(release,indent=2))
    return release

if __name__=='__main__':prepare(ROOT/'configs/dataset-v4-wice-evidence-massive.json')
