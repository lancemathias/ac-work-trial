"""Inspect archived residual logit corrections without loading a backbone or tensors."""
import collections
import json
import math
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['e6-zero-output-init-v31-seed20261014', 'experiment-8-zero-output-init-v31-10epochs-seed20261014']

def softmax(x):
    values=[math.exp(v-max(x)) for v in x];total=sum(values)
    return [v/total for v in values]

def quantiles(values):
    values=sorted(values)
    return {name:values[round(q*(len(values)-1))] for name,q in [('p10',.1),('median',.5),('p90',.9),('p99',.99)]} if values else None

def summarize(rows):
    result={'examples':len(rows),'option_counts':dict(collections.Counter(r['count'] for r in rows))}
    for name in ['range','rms','tv','max_probability_change','effective_support_fraction','top1_energy','top2_energy']:
        result[name]=quantiles([r[name] for r in rows])
    for threshold in [.01,.1,.5,1.]:
        result[f'centered_entries_abs_below_{threshold}']=sum(sum(abs(v)<threshold for v in r['correction']) for r in rows)/sum(r['count'] for r in rows)
    for threshold in [.1,.5,1.,2.]:
        result[f'examples_range_below_{threshold}']=sum(r['range']<threshold for r in rows)/len(rows)
        result[f'examples_max_abs_below_{threshold}']=sum(max(map(abs,r['correction']))<threshold for r in rows)/len(rows)
    for threshold in [.01,.05,.1,.2]:
        result[f'examples_tv_below_{threshold}']=sum(r['tv']<threshold for r in rows)/len(rows)
    if len(result['option_counts']) == 1:
        n=rows[0]['count']
        mean=[statistics.mean(r['correction'][i] for r in rows) for i in range(n)]
        total=sum(sum(v*v for v in r['correction']) for r in rows)
        result['mean_slot_correction']=mean
        result['fraction_energy_explained_by_constant_slot_vector']=len(rows)*sum(v*v for v in mean)/total if total else 0.
    result['argmax_flip_rate']=sum(r['flip'] for r in rows)/len(rows)
    result['correct_to_wrong']=sum(r['base_correct'] and not r['trained_correct'] for r in rows)
    result['wrong_to_correct']=sum(not r['base_correct'] and r['trained_correct'] for r in rows)
    result['both_correct']=sum(r['base_correct'] and r['trained_correct'] for r in rows)
    result['both_wrong']=sum(not r['base_correct'] and not r['trained_correct'] for r in rows)
    return result

def analyze(run):
    with tarfile.open(ROOT/'outputs/model-artifacts'/f'{run}.tar.gz') as archive:
        def read(name):return archive.extractfile(run+'/'+name).read().decode()
        def lines(name):return {r['id']:r for r in map(json.loads,read(name).splitlines())}
        trained=lines('test-native/logits.jsonl');base=lines('base-test-logits.jsonl')
        scored=lines('test-native/raw-predictions.jsonl');base_scored=lines('base-test-native/raw-predictions.jsonl')
        rows=[]
        for key,t in trained.items():
            b=base[key];assert t['option_ids']==b['option_ids']
            delta=[a-v for a,v in zip(t['logits'],b['logits'],strict=True)]
            mean=statistics.mean(delta);centered=[v-mean for v in delta]
            energy=sorted([v*v for v in centered],reverse=True);total=sum(energy)
            p,q=softmax(b['logits']),softmax(t['logits'])
            n=len(delta)
            rows.append({'id':key,'suite':scored[key]['suite'],'count':n,'correction':centered,'range':max(delta)-min(delta),'rms':math.sqrt(total/n),
                'tv':sum(abs(a-v) for a,v in zip(p,q))/2,'max_probability_change':max(abs(a-v) for a,v in zip(p,q)),
                'effective_support_fraction':sum(map(abs,centered))**2/total/n if total else 0.,
                'top1_energy':energy[0]/total if total else 0.,'top2_energy':sum(energy[:2])/total if total else 0.,
                'flip':max(range(n),key=lambda i:t['logits'][i])!=max(range(n),key=lambda i:b['logits'][i]),
                'base_correct':base_scored[key]['correct'],'trained_correct':scored[key]['correct']})
        result={'run':run,'best_epoch':json.loads(read('best.json'))['epoch'],
            'definition':'raw trained minus matched base logits; remove per-example mean; valid options only; uncalibrated probabilities',
            'all':summarize(rows),'by_option_count':{str(n):summarize([r for r in rows if r['count']==n]) for n in sorted({r['count'] for r in rows})},
            'by_suite':{s:summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})}}
        return result,rows

def main():
    out=ROOT/'outputs/analysis/experiment-8-corrections';out.mkdir(parents=True,exist_ok=True)
    for run in RUNS:
        result,rows=analyze(run)
        (out/(run+'-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
        (out/(run+'-per-example.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(json.dumps({'run':run,'flip_rate':result['all']['argmax_flip_rate'],'median_correction_range':result['all']['range']['median']}))

if __name__=='__main__':main()
