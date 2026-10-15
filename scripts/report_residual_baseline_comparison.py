"""Compare cached residual heads with the shared-panel baselines and flag collapse."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def main():
 vendor=read('outputs/baselines-20261014/metrics.json')
 columns={name:vendor[key] for name,key in [('Jev','jev'),('PPLX','pplx'),('CLEF','clef'),('Flash','clef-flash')]}
 for name,path in [('E4','outputs/test/e4-fast-accuracy-v3-seed20261014-test/metrics.json'),('E5','outputs/test/e5-fast-accuracy-bidir-v3-seed20261014-test/metrics.json'),('Readout','outputs/test/readout-fast-qwen38-27b-v1/metrics.json'),('Readout + rev','outputs/test/readout-native-reverse-pool-v1/metrics.json')]:columns[name]=read(path)['suites']
 columns['Fixed v3']=read('outputs/autoresearch/fixed-readout-cpu-recovery-result.json')['summary']['suites']
 columns['Permuted v3.1']=read('outputs/autoresearch/v31-readout-result.json')['test']['suites']
 suites=list(columns['Fixed v3'])
 values={name:{} for name in columns}
 for name,metrics in columns.items():
  for suite in suites:
   m=metrics[suite]
   values[name][suite]=statistics.mean(x['coverage_adjusted_accuracy'] for x in m['subtasks'].values()) if suite=='BBH' else m['coverage_adjusted_accuracy']
  values[name]['Macro']=statistics.mean(values[name][suite] for suite in suites)
 flags={('Flash','RAGTruth'):'†',('E4','ContractNLI'):'†',('E5','RAGTruth'):'‡',('Readout','RAGTruth'):'‡',('Fixed v3','RAGTruth'):'†',('Permuted v3.1','RAGTruth'):'‡',('Fixed v3','BBH'):'§',('Permuted v3.1','BBH'):'§'}
 labels={'FinancialPhraseBank':'Finance','JevBench_public_hard':'JevBench hard','TruthfulQA_binary':'TruthfulQA'}
 lines=['# Residual readouts versus baselines','', 'Accuracy (%) on the same 2,111-example, 11-suite panel. Missing predictions count as wrong; BBH averages its sampled subtasks, and Macro equally weights the 11 suites.', '', 'Fixed/permuted describe training data; both heads are evaluated in native order. Readout + rev pools semantically aligned native/reversed probabilities at inference. E4 is optimized native-causal joint-head training; E5 is its partial-bidirectional counterpart.','', '| Benchmark | '+' | '.join(columns)+' |','|---|'+'---:|'*len(columns)]
 for suite in suites+['Macro']:
  cells=[f'{100*values[name][suite]:.2f}'+flags.get((name,suite),'') for name in columns]
  if suite=='Macro':cells=[f'**{x}**' for x in cells]
  lines.append('| '+labels.get(suite,suite)+' | '+' | '.join(cells)+' |')
 lines += ['', '† Strong collapse/class-elimination evidence. ‡ Strong prediction skew worth investigating. § Small BBH subtask flags, not whole-suite collapse.','',
 '## Collapse diagnostics','',
 '- RAGTruth fixed: True on 199/200 (99.5%), despite only 68/200 True golds; False recall 1/132 (0.76%). Strong collapse.',
 '- RAGTruth permuted: True on 167/200 (83.5%); False recall 28/132 (21.21%). Improved but strongly skewed, and 45.5% accuracy remains below the 66% always-False baseline.',
 '- RAGTruth native readout: False on 186/200 (93%); True recall 13/68 (19.12%). Native + reverse: False on 123/200 (61.5%); True/False recalls 72.06%/78.79%. Pooling substantially reduces this bias.',
 '- RAGTruth E5: True on 182/200 (91%); False recall 16/132 (12.12%).',
 '- RAGTruth CLEF Flash: False on 198/200 (99%); True recall 0/68. No other whole-suite >=90% dominance or zero recall on a class with at least 10 gold examples was found for Jev, PPLX, or CLEF.',
 '- ContractNLI E4: no NotMentioned predictions; 0/83 recall. This class elimination is absent from both new heads.',
 '- BBH tracking_shuffled_objects_five_objects: both new heads predict A on all 8 examples, with 2/8 correct. Fixed also predicts No on all 9 navigate examples (2/9 correct); permuted recovers navigate to 9/9. Sports-understanding is all-no in both, but gold is already 7/8 no. Small slice sizes limit these conclusions.',
 '- No other whole-benchmark >=95% dominant-label flag in the new heads. Circa improves largely through plain-Yes recall, 39/88 to 86/88; fixed overpredicts Probably yes (67 predictions for 10 golds). Rare Circa labels still have zero recall (support only 1–2).','',
 '## Interpretation','',
 'Permutation augmentation raises Macro 70.42% → 75.91% (+5.49 pp), but remains below pretrained Readout (77.14%) and Readout + rev (79.49%). Gains are largest on Circa (+20 pp), Finance (+11 pp), and RAGTruth (+11 pp); the latter remains a failure. The same five-epoch recipe yields 920 updates on v3.1 versus 490 on v3, and doubles choice-view weighting relative to noul. This one-seed contrast does not isolate permutation augmentation from update count and weighting.','']
 (ROOT/'outputs/autoresearch/residual-baseline-comparison.md').write_text('\n'.join(lines))
 (ROOT/'outputs/autoresearch/residual-baseline-comparison.json').write_text(json.dumps({'coverage_adjusted_accuracy':values,'flags':[{ 'model':k[0],'suite':k[1],'flag':v} for k,v in flags.items()]},indent=2)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':main()
