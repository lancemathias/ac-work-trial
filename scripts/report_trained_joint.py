"""Verify and report calibrated E1 predictions on the existing sampled panel."""
import argparse
import json
import math
import statistics
from pathlib import Path
from run_baselines import ROOT,load_panel,metrics,score,read_jsonl,sha,write_json,accuracy_cell


def report(folder,table=None):
    folder=Path(folder)
    cfg=json.loads((folder/'config.json').read_text())
    panel=json.loads((ROOT/'data/manifests/baseline-panel.json').read_text())
    if cfg['panel_sha256']!=sha(ROOT/'data/manifests/baseline-panel.json'):raise ValueError('Panel changed')
    receipt=json.loads((folder/'completion.json').read_text())
    if receipt['status']!='complete' or not receipt['parameters_unchanged']:raise ValueError('Evaluation incomplete')
    originals={r['id']:r for r in load_panel()}
    rows=read_jsonl(folder/'calibrated-predictions.jsonl')
    if len(rows)!=len(originals) or {r['id'] for r in rows}!=set(originals):raise ValueError('Missing/duplicate/unknown IDs')
    for row in rows:
        original=originals[row['id']]
        if row['suite']!=original['suite'] or type(row['label'])!=type(original['label']) or row['label']!=original['label']:
            raise ValueError('Provenance mismatch')
        if row['status']=='ok':
            rebuilt=score(original,row['response'])
            if row['prediction']!=rebuilt['prediction'] or any(not math.isclose(row[k],rebuilt[k],rel_tol=1e-12,abs_tol=1e-12) for k in ['nll','brier']):
                raise ValueError('Stored score mismatch')
    suites={}
    for name,spec in panel['suites'].items():
        selected=[r for r in rows if r['suite']==name]
        m=metrics(selected,spec['count'])
        if name=='BBH':
            m['subtasks']={s:metrics([r for r in selected if r['subtask']==s],n) for s,n in spec['subtasks'].items()}
            m['accuracy']=statistics.mean(x['accuracy'] for x in m['subtasks'].values())
            m['coverage_adjusted_accuracy']=statistics.mean(x['coverage_adjusted_accuracy'] for x in m['subtasks'].values())
        suites[name]=m
    result=dict(temperature=cfg['temperature'],checkpoint_sha256=cfg['checkpoint_sha256'],suites=suites,
        macro_accuracy=statistics.mean(m['accuracy'] for m in suites.values()),
        macro_coverage_adjusted_accuracy=statistics.mean(m['coverage_adjusted_accuracy'] for m in suites.values()),
        macro_nll=statistics.mean(m['nll'] for m in suites.values()),
        macro_brier=statistics.mean(m['brier'] for m in suites.values()),
        macro_ece_15=statistics.mean(m['ece_15'] for m in suites.values()))
    write_json(folder/'metrics.json',result)
    lines=['# E1 calibrated benchmark results','',f"Locked temperature: **T = {cfg['temperature']:.8g}**, fitted on the separate training-source calibration split.",
        '', 'Existing frozen 2,111-example / 11-task sampled panel; one selected checkpoint, no benchmark fitting.',
        '', '| Benchmark | Valid / expected | Accuracy | Misses wrong | NLL | Brier | ECE (15 bins) |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in suites.items():
        lines.append(f"| {name} | {m['valid']}/{m['expected']} | {m['accuracy']:.2%} | {m['coverage_adjusted_accuracy']:.2%} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece_15']:.4f} |")
    lines+= [f"| **Macro** | | **{result['macro_accuracy']:.2%}** | **{result['macro_coverage_adjusted_accuracy']:.2%}** | {result['macro_nll']:.4f} | {result['macro_brier']:.4f} | {result['macro_ece_15']:.4f} |",'',
        'Accuracy macro gives each suite equal weight and averages BBH subtasks first. Probability metrics use valid examples within each suite, then average the 11 suites; BBH probability metrics are example-weighted. Missing outputs count as wrong only in the misses-wrong column. This is the sampled baseline panel, not a claim to reproduce vendor test sets.','']
    (folder/'results.md').write_text('\n'.join(lines))
    if table:
        path=Path(table);text=path.read_text().splitlines();inside=False
        label='E1 Qwen3.5-9B calibrated'
        for i,line in enumerate(text):
            if line.startswith('| Benchmark |'):
                inside=True
                if label not in line:text[i]=line[:-1]+f' {label} |'
            elif inside and line.startswith('|---'):
                if len(line.split('|'))<len(text[i-1].split('|')):text[i]=line+'---:|'
            elif inside and line.startswith('|'):
                cells=[x.strip() for x in line.strip('|').split('|')];name=cells[0]
                m=suites.get(name)
                if m:value=accuracy_cell(m['accuracy'],m['coverage_adjusted_accuracy'],m['valid'],m['expected'])
                elif name.startswith('**Macro'):value=f"{result['macro_accuracy']:.2%} / {result['macro_coverage_adjusted_accuracy']:.2%}"
                else:continue
                header=next(x for x in text if x.startswith('| Benchmark |'))
                count=len(header.strip('|').split('|'))
                text[i]='| '+' | '.join(cells[:count-1]+[value])+' |'
            elif inside:break
        path.write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='suites'},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--table');a=p.parse_args();report(a.folder,a.table)
