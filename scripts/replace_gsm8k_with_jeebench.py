"""Replace synthetic GSM8K choices with native JEEBench math MCQs, exhaustively screened."""
import collections, hashlib, json, re, zipfile
from pathlib import Path
from generate_typed_training import convert, canonical, validate
from finalize_training_sample import norm, shingles, texts
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 raw=ROOT/'data/training/raw/JEEBench'; rev=json.loads((raw/'revision.json').read_text())['revision']
 native=json.loads(zipfile.ZipFile(raw/'data.zip').read('data/dataset.json'))
 candidates=[r for r in native if r['subject']=='math' and r['type']=='MCQ']
 index=collections.defaultdict(set);exact=collections.defaultdict(set);lengths={};files=[]
 expected=json.loads((ROOT/'data/manifests/benchmark-preprocessing.json').read_text())['suites']
 for p in sorted(Path('/Users/bwt/ac-worktrial/data/processed').glob('*.jsonl')):
  assert sha(p)==expected[p.stem]['sha256'];files.append({'file':p.name,'sha256':sha(p)})
  for line in p.open():
   r=json.loads(line)
   for text in texts(r['state']):
    w=norm(text)
    if len(w)<8:continue
    h=hashlib.sha256(' '.join(w).encode()).hexdigest();exact[h].add(p.stem)
    if h in lengths:continue
    ss=shingles(w);lengths[h]=len(ss)
    for s in ss:index[s].add(h)
 kept=[];reject=[]
 for r in candidates:
  rid=f"JEEBench:{r['description']}:{r['index']}";q=r['question'];matches=list(re.finditer(r'[\(\[]([A-D])[\)\]]\s*',q))
  if [m.group(1) for m in matches]!=list('ABCD'):
   reject.append({'id':rid,'reason':'Could not parse exactly four native choices'});continue
  stem=q[:matches[0].start()].strip();options={m.group(1):q[m.end():matches[i+1].start() if i+1<len(matches) else len(q)].strip() for i,m in enumerate(matches)}
  hits=[]
  for text in [stem,q]:
   w=norm(text);h=hashlib.sha256(' '.join(w).encode()).hexdigest()
   if h in exact:hits.append({'kind':'exact','suites':sorted(exact[h])})
   ss=shingles(w)
   if len(ss)>=5:
    counts=collections.Counter(e for s in ss for e in index.get(s,()))
    for e,n in counts.items():
     if n>=5 and n/min(len(ss),lengths[e])>=.8:hits.append({'kind':'16-word containment','suites':sorted(exact[e])})
  if hits:reject.append({'id':rid,'reason':'benchmark overlap','hits':hits});continue
  row={'id':rid,'source':'JEEBench','family_id':rid,'input':{'state':'','question':stem,'options':options},'target':r['gold'],'source_split':'official evaluation release repurposed for training; not evaluated as JEEBench','source_revision':rev,'provenance':{'repository':'https://github.com/dair-iitd/jeebench','archive_sha256':sha(raw/'data.zip'),'exam':r['description'],'index':r['index']}}
  typed=convert(row);validate(typed);kept.append(typed)
 source=ROOT/'data/training/typed-4500-no-helpsteer2-banking77.jsonl'
 old=[json.loads(l) for l in source.open()];rows=[r for r in old if r['source']!='GSM8K']+kept
 assert len({r['id'] for r in rows})==len(rows)
 for r in rows:validate(r)
 target=ROOT/f'data/training/typed-{len(rows)}-jeebench.jsonl';target.write_text(''.join(canonical(r)+'\n' for r in rows))
 hist=collections.Counter(len(r['model_payload']['schema']['properties'][f]['enum']) for r in rows for f,t in r['field_types'].items() if t=='choice')
 report={'source_revision':rev,'archive_sha256':sha(raw/'data.zip'),'input_sha256':sha(source),'output_sha256':sha(target),'removed_GSM8K':250,'eligible_JEEBench_math_MCQ':len(candidates),'accepted_JEEBench':len(kept),'rejected':reject,'records':len(rows),'question_types':dict(collections.Counter(t for r in rows for t in r['field_types'].values())),'choice_option_histogram':dict(sorted(hist.items())),'seed':20261016,'selection':'All eligible records, official order; no resampling or duplication. Existing schema permutation seed unchanged.','screen':'Exact normalized stem/full question and 80% containment of 16-word shingles, at least 5 shared shingles; compared against all benchmark state text fields.','evaluation_files':files,'lineage_assessment':'JEE Advanced 2016-2023 is not a documented LiveBench source (recent AMC/AIME/SMC/olympiad competitions and AMPS). No established direct benchmark lineage found; independent reuse in MMLU-Pro or other corpora and paraphrases remain possible. This is a local screening pass, not a contamination-free guarantee.','artifact_note':'Existing Modal jobs/caches retain original dataset identity; no forward cache exists for new JEEBench rows.'}
 (ROOT/'outputs/jeebench-replacement-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 (ROOT/'configs/active-training-dataset.json').write_text(json.dumps({'dataset_path':str(target.relative_to(ROOT)),'dataset_sha256':sha(target),'records':len(rows),'excluded_sources':['HelpSteer2','BANKING77','GSM8K'],'replacement_source':'JEEBench','derivation_script':'scripts/replace_gsm8k_with_jeebench.py','audit':'outputs/jeebench-replacement-audit.json'},indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='evaluation_files'},indent=2))
if __name__=='__main__':main()
