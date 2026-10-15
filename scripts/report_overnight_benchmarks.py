"""Reproduce tables from checksum-verified completed metrics; no model inference."""
import csv,hashlib,json,statistics,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SUITES=['BBH','Belebele','Circa','ContractNLI','FinancialPhraseBank','JevBench_public_hard','JudgeBench','RAGTruth','TabFact','TruthfulQA_binary','WinoGrande']
LABELS={'FinancialPhraseBank':'Finance','JevBench_public_hard':'JevBench hard','TruthfulQA_binary':'TruthfulQA'}

def main():
 run_meta={};scores={};sources={}
 baseline=ROOT/'outputs/baselines-20261014';raw=(baseline/'metrics.json').read_bytes();index=json.loads((baseline/'artifact-index.json').read_text())
 if hashlib.sha256(raw).hexdigest()!=index['files']['metrics.json']['sha256']:raise ValueError('Baseline metrics hash changed')
 data=json.loads(raw)
 for key,label in [('jev','Jev 1.13'),('pplx','PPLX v1'),('clef','CLEF'),('clef-flash','CLEF Flash')]:
  d=data[key];values={s:d[s].get('task_macro_coverage_adjusted_accuracy',d[s]['coverage_adjusted_accuracy']) for s in SUITES};values['Composite']=d['overall']['coverage_adjusted_suite_macro_accuracy'];scores[label]=values;sources[label]={'path':'outputs/baselines-20261014/metrics.json','sha256':hashlib.sha256(raw).hexdigest(),'model_key':key};run_meta[label]='Released baseline'
 runs=[('E1','e1-qwen35-9b-test-d4bd899'),('E2','e2-qwen35-9b-v2-test-1883787'),('E3','e3-medium-v3-seed20261014-e7dd4bb-test'),('E4 fallback','e4-accuracy-v3-seed20261014-test'),('E4 optimized','e4-fast-accuracy-v3-seed20261014-test'),('E5 bidirectional','e5-fast-accuracy-bidir-v3-seed20261014-test'),('Native 27B','readout-fast-qwen38-27b-v1'),('Native + reverse','readout-native-reverse-pool-v1'),('Source control','ab1-math-accuracy-seed20261014-control-1-test'),('RuleTaker','ab1-math-accuracy-seed20261014-ruletaker-test'),('Aegis2','ab1-math-accuracy-seed20261014-aegis2-test'),('CivilComments','ab1-math-accuracy-seed20261014-civilcomments-test'),('SNLI + MultiNLI','ab1-math-accuracy-seed20261014-snli-multinli-test'),('MASSIVE','ab1-math-accuracy-seed20261014-massive-test'),('BANKING77','ab1-math-accuracy-seed20261014-banking77-test'),('RuleTaker repaired','ruletaker-cwa-explicit-math-seed20261014-test'),('E5 + native mixture','cached-e5-readout-equal-mixture-v1')]
 for label,name in runs:
  folder=ROOT/'outputs/test'/name
  if not (folder/'checksums.json').exists():folder=ROOT/'outputs/model-artifacts'/name
  raw=(folder/'metrics.json').read_bytes();checks=json.loads((folder/'checksums.json').read_text())
  derived=None
  if 'metrics.json' in checks:
   if hashlib.sha256(raw).hexdigest()!=checks['metrics.json']:raise ValueError('Candidate metric changed: '+name)
  else:
   # Early reports were derived after the original trace seal. Rebuild their
   # points from original checksum-verified, locally rescored predictions.
   import collections
   from compare_predictions import verified,aggregate
   original=ROOT/'outputs/model-artifacts'/name;rows=verified(original);strata=collections.defaultdict(list)
   for r in rows:strata[(r['suite'],r.get('subtask') if r['suite']=='BBH' else '')].append(int(r['status']=='ok' and r.get('correct') is True))
   total,per_suite=aggregate({k:statistics.mean(v) for k,v in strata.items()});derived=dict(per_suite,Composite=total)
   sources[label]={'prediction_path':str((original/'calibrated-predictions.jsonl').relative_to(ROOT)),'prediction_sha256':hashlib.sha256((original/'calibrated-predictions.jsonl').read_bytes()).hexdigest(),'verification':'Checksum-verified original predictions rescored against fixed panel'}
  cfg=json.loads((folder/'config.json').read_text())
  if cfg['panel_sha256']!=index['panel_sha256']:raise ValueError('Different panel: '+name)
  d=json.loads(raw);values={s:d['suites'][s]['coverage_adjusted_accuracy'] for s in SUITES};values['Composite']=d['macro_coverage_adjusted_accuracy']
  if derived is not None:
   if any(abs(values[s]-derived[s])>1e-12 for s in SUITES+['Composite']):raise ValueError('Historical metrics do not match rescored predictions')
   values=derived
  scores[label]=values;sources.setdefault(label,{'path':str((folder/'metrics.json').relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'run_id':name});run_meta[label]='Historical context' if label in ['E1','E2','E3'] else 'Completed overnight/morning candidate'
 for label,values in scores.items():
  if abs(statistics.mean(values[s] for s in SUITES)-values['Composite'])>1e-12:raise ValueError('Composite averaging mismatch: '+label)
 headline=['Jev 1.13','PPLX v1','CLEF','CLEF Flash','E4 optimized','E5 bidirectional','Native 27B','Native + reverse']
 sources_cols=['Jev 1.13','Source control','RuleTaker','Aegis2','CivilComments','SNLI + MultiNLI','MASSIVE','BANKING77','RuleTaker repaired']
 history=['Jev 1.13','E1','E2','E3','E4 fallback','E5 + native mixture']
 def table(columns):
  lines=['| Benchmark | '+' | '.join(columns)+' |','| --- | '+' | '.join(['---:']*len(columns))+' |']
  for suite in SUITES+['Composite']:
   cells=[f'{100*scores[c][suite]:.2f}' for c in columns];name=LABELS.get(suite,suite)
   if suite=='Composite':name='**Composite**';cells=['**'+x+'**' for x in cells]
   lines.append('| '+name+' | '+' | '.join(cells)+' |')
  return '\n'.join(lines)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();doc=['# Overnight benchmark scores','',f'Generated {now}. Accuracy percentages on the shared fixed 2,111-example panel; 200 per benchmark except 111 JevBench hard. Misses count as wrong. BBH is averaged across its included subtasks; the composite equally weights the 11 benchmark scores.','', '## Baselines and headline overnight runs','',table(headline),'', 'E4 optimized is the accuracy-selected 27B trained head with native causality. E5 removes the full-attention causal mask only; recurrent/convolution paths remain causal. Native 27B uses the frozen pretrained answer-letter readout. Native + reverse averages semantically aligned probabilities from both option orders; it is the best completed research run.','', '## Source ablations beside Jev','',table(sources_cols),'', 'The source screen uses frozen 9B, the same Medium recipe, 6,512 training decisions, 510 updates and fixed dev/calibration sets. Each addition has 250 decisions; the control adds 250 fresh examples from existing sources. Repaired RuleTaker changes only its instructions and features. These are not matched backbone/recipe comparisons against the 27B headline runs.','', '## Earlier runs and rejected mixture','',table(history),'', 'E1–E3 are historical context, not overnight reruns. E4 fallback uses the earlier fallback numerical runtime; compare E5 against E4 optimized for the attention ablation. The equal E5/native mixture passed dev selection but lost test accuracy and was rejected. E2 head-size ablations have dev measurements, not completed comparable 11-benchmark test columns.','', '## Pending final run and interpretation','', 'The final bounded-thinking run is excluded until its full 11-benchmark test evaluation completes. A completed dev result alone is not a benchmark score. The user requested this be the last GPU run; automatic export and GPU release are armed, and no follow-ups will launch.','', 'These are sampled-panel exploratory point estimates. They do not establish full-dataset superiority, seed robustness or independent confirmation. CLEF and CLEF Flash each miss one ContractNLI input; the table counts it wrong consistently. Temperatures were fitted only on permitted nonbenchmark calibration examples for trained/research candidates; released baselines retain their original probabilities.','', '## Evidence','', '- [Machine-readable scores and input SHA-256 hashes](../outputs/autoresearch/overnight-benchmark-scores.json).','- [Wide CSV with every included run](../outputs/autoresearch/overnight-benchmark-scores.csv).','- [Baseline measurements](../outputs/baselines-20261014/results.md), [source ledger](source-screen-ledger.md), and [issues hypotheses and fixes](overnight-research-log.md).','']
 (ROOT/'docs/overnight-benchmark-scores.md').write_text('\n'.join(doc))
 payload={'generated_utc':now,'metric':'coverage_adjusted_accuracy','panel_sha256':index['panel_sha256'],'BBH_aggregation':'mean of included subtasks','composite_aggregation':'equal mean of11benchmarks','scores':scores,'input_artifacts':sources,'run_context':run_meta,'thinking_status':'pending_not_a_completed_test_result'}
 (ROOT/'outputs/autoresearch/overnight-benchmark-scores.json').write_text(json.dumps(payload,indent=2)+'\n')
 with (ROOT/'outputs/autoresearch/overnight-benchmark-scores.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['Benchmark',*scores]);writer.writerows([[s,*[100*values[s] for values in scores.values()]] for s in SUITES+['Composite']])
 print(table(headline));print('Verified',len(scores),'runs, common panel, metric checksums and all composite means.')
if __name__=='__main__':main()
