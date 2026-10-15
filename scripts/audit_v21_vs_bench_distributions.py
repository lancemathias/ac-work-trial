"""Compare dataset distributions and effective weights in our local 11-suite score."""
import collections,json,pathlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];BENCH=pathlib.Path('/Users/bwt/ac-worktrial/data/processed')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(rows,weights=None):
 weights=weights or [1.]*len(rows);types=collections.Counter();options=collections.Counter()
 for row,w in zip(rows,weights):
  q=row['question'];types[q['type']]+=w
  if q['type']=='choice':options[len(q['criteria'])]+=w
 total=sum(types.values());choice=types['choice']
 return {'records':len(rows),'type_counts_or_weight':dict(types),'type_percent':{k:100*types[k]/total for k in ['noul','choice','score']},'choice_option_counts_or_weight':dict(sorted(options.items())),'choice_option_percent':{k:100*v/choice for k,v in sorted(options.items())},'choice_option_mean':sum(k*v for k,v in options.items())/choice}
def effective_weights(rows):
 suites=collections.Counter(r['suite'] for r in rows);sub=collections.Counter(r['source']['subtask'] for r in rows if r['suite']=='BBH')
 return [(1/len(suites)/len(sub)/sub[r['source']['subtask']] if r['suite']=='BBH' else 1/len(suites)/suites[r['suite']]) for r in rows]
def main():
 dataset=ROOT/'data/training/train-5053-six-sources-quality-preference.jsonl';train=[json.loads(l) for l in dataset.open()];bench=[];files=[]
 for p in sorted(BENCH.glob('*.jsonl')):bench.extend(json.loads(l) for l in p.open());files.append({'file':p.name,'sha256':sha(p)})
 panelpath=pathlib.Path('/Users/bwt/ac-worktrial/data/manifests/baseline-panel.json');panel=json.loads(panelpath.read_text());byid={r['id']:r for r in bench};selected=[byid[i] for s in panel['suites'].values() for i in s['ids']]
 report={'dataset_version':'v2.1','dataset_sha256':sha(dataset),'train':summarize(train),'benchmark_pooled':summarize(bench),'benchmark_full_score_weighted':summarize(bench,effective_weights(bench)),'benchmark_actual_panel_score_weighted':summarize(selected,effective_weights(selected)),'benchmark_files':files,'panel_sha256':sha(panelpath),'weighting':'Our implemented baseline score: equal 1/11 per suite; within BBH equal weight per represented subtask, then equal records per subtask; within other suites equal records. Choice option percentages are conditional on Choice score mass. Not verified Perplexity official weights.','limits':['Full benchmark pool contains known malformed single-option BBH:snarks:88, retained for counting to match current stored corpora.','Option count counts criteria keys, not embedded original MCQs inside JudgeBench response comparisons.']}
 out=ROOT/'outputs/v21-vs-bench-distribution-audit.json';out.write_text(json.dumps(report,indent=2)+'\n')
 for k in ['train','benchmark_pooled','benchmark_full_score_weighted','benchmark_actual_panel_score_weighted']:print(k,json.dumps(report[k],indent=2))
if __name__=='__main__':main()
