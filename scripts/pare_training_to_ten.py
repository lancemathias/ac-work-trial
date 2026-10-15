"""Select ten complementary training sources; no resampling or target changes."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SELECTION={
'ANLI':('ContractNLI; BBH formal reasoning','Three-way entailment/contradiction/neutral decisions over premise and hypothesis; short general text rather than contracts.','Same NLI task family as ContractNLI, not an established shared dataset lineage.'),
'SST2':('FinancialPhraseBank','Sentiment classification; film reviews and binary polarity rather than financial text with a neutral class.','Related sentiment task, separate documented corpus.'),
'RACE':('Belebele','Passage plus question and four native answer choices; reading comprehension.','Related format and skill; exam passages rather than FLORES passages.'),
'ReClor':('BBH; JevBench multi_hop and policy reasoning','Four-choice logical inference and argument assessment from a supplied context.','Related reasoning task; not an established shared source.'),
'RuleTaker':('TabFact; ContractNLI; JevBench multi_hop','Boolean support decisions from supplied facts and explicit rules. No actual tables or contract language.','Related inference skill; synthetic rule corpus rather than benchmark documents.'),
'MultiRC':('RAGTruth; TruthfulQA binary; Belebele','Boolean correctness of a supplied answer relative to a passage. Closer evidence-based decision structure than free response.','Related answer-checking task; does not reproduce full-response hallucination detection or global truthfulness.'),
'WIQA':('BBH causal_judgement; JevBench multi_hop','Three-choice causal effects of changes to procedural descriptions.','Related causal reasoning task; different corpus and question construction.'),
'SciQ':('JudgeBench science; TruthfulQA; Belebele','Four-choice science questions with supporting text. Science domain coverage; support often directly supplies answer.','Related science domain, not the JudgeBench response-comparison task. Distractors can be weak.'),
'MASSIVE':('JevBench routing_hard','Schema-conditioned intent classification and routing over an intent taxonomy.','Related routing task; 60 options and short utterances differ substantially from policy routing benchmarks.'),
'JEEBench':('JudgeBench LiveBench-math and MMLU-Pro math','Difficult native four-choice algebra, calculus, geometry and probability problems.','Related math domain; direct answer choice rather than judging two worked solutions. Locally overlap-screened; shared upstream reuse remains uncertain.')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=ROOT/'data/training/train-4303-benchmark-format.jsonl'
 lines=source.read_bytes().splitlines(keepends=True);keep=[l for l in lines if json.loads(l)['suite'] in SELECTION]
 rows=[json.loads(l) for l in keep];assert len(rows)==2303 and len({r['suite'] for r in rows})==10
 target=ROOT/'data/training/train-2303-ten-sources.jsonl';target.write_bytes(b''.join(keep))
 descriptions=[]
 for s,(bench,similarity,relation) in SELECTION.items():
  selected=[r for r in rows if r['suite']==s];descriptions.append({'source':s,'records':len(selected),'question_types':dict(collections.Counter(r['question']['type'] for r in selected)),'related_benchmarks':bench,'similarity_and_difference':similarity,'relationship':relation})
 report={'input':str(source.relative_to(ROOT)),'input_sha256':sha(source),'dataset':str(target.relative_to(ROOT)),'dataset_sha256':sha(target),'records':len(rows),'sources':descriptions,'removed_sources':sorted({json.loads(l)['suite'] for l in lines}-set(SELECTION)),'question_types':dict(collections.Counter(r['question']['type'] for r in rows)),'randomness':'None. Source filtering only; row bytes, order, option keys and labels unchanged.','gaps':['No direct conversational indirect-answer classification equivalent to Circa.','No native table verification equivalent to TabFact.','No worked-solution pair evaluation equivalent to JudgeBench.','No WinoGrande-specific pronoun resolution training source.','No Score supervision.','Option counts lack eight-choice questions; MASSIVE retains a 60-choice tail.'],'lineage_limit':'Relationships above are task/domain analogies, not evidence of shared rows. Prior local heuristic screens do not establish universal contamination freedom.','cache_status':'Existing regeneration targets previous 4303-record version; retained rows are byte-identical and compatible with its request hashes. Use only selected IDs; a new forward pass is unnecessary for this source-only reduction.'}
 (ROOT/'outputs/ten-source-selection-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 p=ROOT/'configs/active-training-dataset.json';c=json.loads(p.read_text());c.update({'dataset_path':str(target.relative_to(ROOT)),'dataset_sha256':sha(target),'records':len(rows),'selected_sources':list(SELECTION),'selection_audit':'outputs/ten-source-selection-audit.json','selection_script':'scripts/pare_training_to_ten.py'});p.write_text(json.dumps(c,indent=2)+'\n')
 print(json.dumps({'records':len(rows),'types':report['question_types'],'removed':report['removed_sources'],'sha256':sha(target)},indent=2))
if __name__=='__main__':main()
