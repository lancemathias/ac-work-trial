"""Audit and summarize the completed E9-protocol v4 sweep."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/training/v4-residual-l2-sweep'
result=json.loads((out/'result.json').read_text())
expected=[.03,.1,.3,1.,3.]
assert result['request']['spec']['lambdas']==expected
assert result['request']['initial_state_sha256']=='d94791120cd73a5d63d9520ad16f875100e5bc9ed72fbab1c7a183316e1a6044'
assert result['updates_per_lambda']==3370 and result['total_updates']==16850
rows=[]
for c in result['candidates']:
    records=json.loads((out/c['folder']/'learning-curves.json').read_text())
    assert [r['epoch'] for r in records]==list(range(1,11))
    for r in records:
        assert all(math.isfinite(v) for v in r['online_train'].values())
        assert abs(r['online_train']['objective']-(r['online_train']['nll']+c['lambda']*r['online_train']['residual_l2']))<1e-5
        assert r['train_eval']['decisions']==21524 and r['dev']['decisions']==809
    best=max(records,key=lambda r:(r['dev']['accuracy'],-r['epoch']))
    assert best['epoch']==c['best_epoch'] and best['dev']==c['dev']
    rows.append({'lambda':c['lambda'],'best_epoch':c['best_epoch'],'dev_accuracy':c['dev']['accuracy'],'dev_nll':c['dev']['nll'],'train_accuracy':best['train_eval']['accuracy'],'train_nll':best['train_eval']['nll'],'dev_residual_rms':c['dev']['residual_rms'],'wandb':c['wandb']})
selected=max(result['candidates'],key=lambda c:(c['dev']['accuracy'],-c['dev']['nll'],-c['lambda']))
assert selected==result['winner']
selection=json.loads((out/'selection.json').read_text());assert selection['test_accessed'] is False and selection['winner']==selected
assert sum((out/c['folder']/'test-native').exists() for c in result['candidates'])==1
with (out/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
audit={'status':'passed','grid_exact':True,'complete_epochs':50,'initialization_matches_e9_sha256':True,'updates_per_lambda':3370,'objective_decomposition_valid':True,'earliest_accuracy_ties_valid':True,'global_selection_valid':True,'role_counts_valid':True,'winner_only_test':True}
(out/'postrun-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
lines=['# v4 residual L2 sweep','', 'E9 protocol; 10 epochs per lambda, same saved zero-output initialization and optimizer settings. v4 has 21,524 training views; unchanged dev/calibration and benchmark panel. Metrics/configuration published to W&B; checkpoints preserved locally.','', '| Lambda | Best epoch | Dev accuracy | Train accuracy | Dev NLL | W&B |','|---:|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['lambda']:g} | {r['best_epoch']} | {100*r['dev_accuracy']:.2f}% | {100*r['train_accuracy']:.2f}% | {r['dev_nll']:.4f} | [run]({r['wandb']}) |")
lines.extend(['',f"Selected lambda: **{selected['lambda']:g}**, epoch **{selected['best_epoch']}**, using dev accuracy, then dev NLL, then smaller lambda. Only the selected checkpoint was tested.",'',f"Benchmark macro accuracy: **{100*result['test']['macro_accuracy']:.2f}%**.",'','All 50 epochs log online training CE, residual penalty/objective, full training evaluation and dev metrics. The 10-epoch schedule yields 3,370 optimizer updates per lambda, versus 1,840 in E9 because v4 is larger.'])
e9=json.loads((ROOT/'outputs/training/experiment-9-residual-l2-sweep/summary.json').read_text())['test']
comparison={'v4_test_macro_accuracy':result['test']['macro_accuracy'],'e9_test_macro_accuracy':e9['macro_accuracy'],'difference_pp':100*(result['test']['macro_accuracy']-e9['macro_accuracy']),'ragtruth':{'v4':result['test']['suites']['RAGTruth']['accuracy'],'e9':e9['suites']['RAGTruth']['accuracy']},'qualification':'Matching ten epochs means more optimizer updates with v4; the comparison does not isolate dataset effects.'}
(out/'e9-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
lines.extend(['',f"E9 selected winner: {100*e9['macro_accuracy']:.2f}% test macro accuracy; v4 difference {comparison['difference_pp']:+.2f} percentage points. RAGTruth: {100*comparison['ragtruth']['v4']:.1f}% versus {100*comparison['ragtruth']['e9']:.1f}% in E9. This comparison does not isolate the effect of added data."])
(out/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'audit':audit,'winner':selected,'test':result['test']},indent=2))
