"""Descriptive E8/E9 coverage audit; lexical screens do not establish semantic coverage."""
import collections
import hashlib
import json
import re
import tarfile
from pathlib import Path
from run_baselines import load_panel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/analysis/experiment-9-coverage'
NEG=re.compile(r'\b(?:not|no|never|missing|absent|absence|without|lack|lacks|lacking|neither|nor|except)\b',re.I)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def length(row):return len(json.dumps(row['state'],ensure_ascii=False))
def quantiles(rows):
    values=sorted(map(length,rows))
    return {name:values[round(q*(len(values)-1))] for name,q in [('median',.5),('p90',.9),('max',1.)]}

def main():
    path=ROOT/'work/autoresearch/data/training/dataset-v3.jsonl'
    assert sha(path)=='ef15c46f49192cafe5d35a1b0567e1a33cc13807fb5c2ca4d50e1fdbc6beec81'
    split=ROOT/'work/autoresearch/outputs/model-artifacts/e3-medium-v3-seed20261014-e7dd4bb-splits'
    manifest=json.loads((split/'manifest.json').read_text())
    spec=manifest['splits']['train']['points'];trainpath=split/spec['path'];assert sha(trainpath)==spec['sha256']
    ids={json.loads(x)['id'] for x in trainpath.read_text().splitlines()}
    train=[json.loads(x) for x in path.read_text().splitlines() if json.loads(x)['id'] in ids]
    assert len(train)==6262
    noul=[r for r in train if r['question']['type']=='noul']
    binary=[r for r in train if r['question']['type']=='choice' and len(r['question']['criteria'])==2]
    report={'dataset_sha256':sha(path),'train_split_sha256':sha(trainpath),
      'notes':['Lengths are serialized-state CHARACTERS, not prompt tokens.', 'Negative-word screen uses instructions only; words in evidence/answers may be missed, and hits can be irrelevant. This is not semantic negation annotation.', 'All analysis is post-hoc; no selection or training changes.'],
      'training':{'native_records':len(train),'noul_records':len(noul),'noul_sources':dict(collections.Counter(r['suite'] for r in noul)),
       'noul_labels':dict(collections.Counter(str(r['label']) for r in noul)),
       'noul_affirmative_correctness_predicate':sum(r['question']['instructions'].endswith('Is the candidate answer correct?') for r in noul),
       'noul_instruction_negative_word_hits':sum(bool(NEG.search(r['question']['instructions'])) for r in noul),
       'noul_augmented_view_fraction':len(noul)/11755,
       'binary_choice_records':len(binary),'binary_choice_sources':dict(collections.Counter(r['suite'] for r in binary)),
       'state_chars':quantiles(train),'noul_state_chars':quantiles(noul)}}
    winnerfile=OUT/'winner-test-predictions.jsonl'
    winner={r['id']:r for r in map(json.loads,winnerfile.read_text().splitlines())}
    run='experiment-8-zero-output-init-v31-10epochs-seed20261014'
    with tarfile.open(ROOT/'outputs/model-artifacts'/f'{run}.tar.gz') as archive:
        base={r['id']:r for r in map(json.loads,archive.extractfile(run+'/base-test-native/raw-predictions.jsonl').read().decode().splitlines())}
    rows=load_panel(); assert set(winner)==set(base)=={r['id'] for r in rows}
    report['winner_predictions_sha256']=sha(winnerfile)
    report['test_suites']={}
    for suite in ['RAGTruth','ContractNLI','TabFact','TruthfulQA_binary']:
        selected=[r for r in rows if r['suite']==suite]
        entry={'count':len(selected),'state_chars':quantiles(selected),'instructions':list(dict.fromkeys(r['question']['instructions'] for r in selected)), 'length_slices':{}}
        for label,lo,hi in [('below_3606',0,3606),('3606_to_11874',3606,11875),('above_11874',11875,float('inf'))]:
            subset=[r for r in selected if lo<=length(r)<hi]
            if subset:
                entry['length_slices'][label]={'count':len(subset),'base_accuracy':sum(base[r['id']]['correct'] for r in subset)/len(subset),'winner_accuracy':sum(winner[r['id']]['correct'] for r in subset)/len(subset)}
        report['test_suites'][suite]=entry
    (OUT/'coverage-and-length.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
