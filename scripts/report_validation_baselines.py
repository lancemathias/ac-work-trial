"""Report vendor released probabilities versus E2's calibrated validation outputs."""
import argparse,json,statistics
from pathlib import Path
from run_baselines import read_jsonl,metrics,score,record,write_json,sha
from validation_baselines import load_validation

def report(splits,jev,clef,e2,out):
 rows,meta=load_validation(splits);byid={r['id']:r for r in rows};predictions={}
 for name,folder in [('Jev 1.13.0',jev),('CLEF-flash',clef)]:
  folder=Path(folder);cfg=json.loads((folder/'config.json').read_text())
  if cfg['panel_sha256']!=meta['panel_sha256']:raise ValueError('Mismatched validation panel')
  rs=read_jsonl(folder/'predictions.jsonl')
  if len(rs)!=len(rows) or {r['id'] for r in rs}!=set(byid):raise ValueError('Incomplete/duplicate validation predictions')
  predictions[name]=rs
 traces=read_jsonl(Path(e2)/'dev-traces.jsonl');rs=[]
 for t in traces:
  row=byid[t['id']];q=row['question'];p=dict(zip(t['option_ids'],t['calibrated_probabilities'],strict=True))
  answer={'type':q['type']}
  if q['type']=='noul':answer['noul']=p['true']
  else:answer['probabilities']=p
  body={'answers':{'q':answer}}
  rs.append(record(row,dict(score(row,body),latency_ms=0)))
 if len(rs)!=len(rows) or {r['id'] for r in rs}!=set(byid):raise ValueError('E2 validation mismatch')
 predictions['E2 calibrated']=rs
 result={'protocol':meta,'models':{}}
 for name,rs in predictions.items():
  result['models'][name]={'overall':metrics(rs,len(rows)),'sources':{s:metrics([r for r in rs if r['suite']==s],sum(r['suite']==s for r in rows)) for s in sorted({r['suite'] for r in rows})}}
 out=Path(out);out.mkdir(parents=True,exist_ok=True);write_json(out/'metrics.json',result)
 lines=['# E2 validation comparison','', 'Same 262-example E2 validation split; no fitting or calibration of vendor models on this split. CLEF-flash uses the pinned released weights; Jev uses API model jev-1.13.0. E2 uses its validation-selected epoch-4 checkpoint and T=1.51914063735723 fitted on separate calibration data. E2 validation was used for checkpoint selection, so it is not an unbiased final test comparison.','', '| Model | Valid / expected | Accuracy (misses wrong) | NLL | Brier | ECE-15 |','|---|---:|---:|---:|---:|---:|']
 for name,m in result['models'].items():
  x=m['overall'];lines.append(f"| {name} | {x['valid']}/{x['expected']} | {x['coverage_adjusted_accuracy']:.2%} | {x['nll']:.4f} | {x['brier']:.4f} | {x['ece_15']:.4f} |")
 lines+=['','| Source | n | CLEF-flash | Jev 1.13.0 | E2 calibrated |','|---|---:|---:|---:|---:|']
 for s in sorted({r['suite'] for r in rows}):
  n=sum(r['suite']==s for r in rows);values=[result['models'][name]['sources'][s]['coverage_adjusted_accuracy'] for name in ['CLEF-flash','Jev 1.13.0','E2 calibrated']]
  lines.append(f"| {s} | {n} | "+' | '.join(f'{v:.2%}' for v in values)+' |')
 lines+=['','Overall metrics weight each validation decision equally; source rows are descriptive, not a source-macro aggregate. NLL/Brier/ECE use valid outputs; accuracy counts missing outputs as wrong. NLL uses a 1e-12 floor. No vendor temperature fitting was performed.','']
 (out/'results.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['splits','jev','clef','e2','out']:p.add_argument('--'+k,required=True)
 a=p.parse_args();report(**vars(a))
