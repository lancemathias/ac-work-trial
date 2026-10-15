"""Deterministic, local typed-schema recasts. No model or network calls."""
from __future__ import annotations
import argparse, collections, decimal, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEED=20261016
RUBRIC_SOURCE='https://arxiv.org/html/2406.08673v1#A7.SS3.SSS1'
RUBRICS={
 'helpfulness':'Overall usefulness for the request. 0: no utility; 1: limited utility; 2: partly meets intent; 3: mostly meets intent; 4: fully meets intent.',
 'correctness':'Accuracy and requested coverage. 0: wholly wrong or irrelevant; 1: largely erroneous or incomplete; 2: mixed accuracy; 3: accurate with small omissions; 4: accurate and complete.',
 'coherence':'Clarity and internal consistency. 0: unintelligible; 1: predominantly confusing; 2: some confusing or inconsistent passages; 3: minor clarity issues; 4: clear and consistent throughout.',
 'complexity':'Language sophistication, not quality. 0: basic everyday language; 1: elementary/middle-school language; 2: high-school vocabulary; 3: university-level specialist language; 4: professional domain expertise.',
 'verbosity':'Detail relative to the request, not quality. 0: minimal concise answer; 1: short with some removable detail; 2: typical adequate length; 3: extended answer; 4: extensive extra detail or wordiness.'
}
EXPECTED_INPUT='7ac75ce3895d89f22cc4ef9095112302454d0df3ff64c5c6cdfe9b6f86f3d80b'
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(seed,rid,field,index): return hashlib.sha256(canonical([seed,rid,field,index]).encode()).hexdigest()
def choice(options,target,seed,rid,field='decision'):
 items=list(options.items()) if isinstance(options,dict) else [(str(i),v) for i,v in enumerate(options)]
 if isinstance(options,list): correct=str(target)
 elif str(target) in options: correct=str(target)
 else:
  matches=[k for k,v in items if v==target]
  if len(matches)!=1: raise ValueError((rid,'unresolved target',target))
  correct=matches[0]
 order=sorted(range(len(items)),key=lambda i:(rank(seed,rid,field,i),i)); mapping={items[i][0]:f'option_{j}' for j,i in enumerate(order)}
 descriptions={mapping[items[i][0]]:str(items[i][1]) for i in order}
 schema={'type':'string','enum':list(descriptions),'description':'Select exactly one allowed option. Options: '+canonical(descriptions)}
 return schema,mapping[correct],{'operation':'choice_permutation','field':field,'original_to_typed':mapping,'option_descriptions':descriptions,'permutation_original_indices':order}
def convert(r,seed=SEED):
 source=r['source']; rid=r['id']; inp=r.get('input',r); state=inp.get('state','');q=inp['question'];options=inp.get('options'); target=r['target']; traces=[]; metadata={}; props={}; gold={}
 if source=='HelpSteer2' and r['source_config']=='ratings':
  for field,value in sorted(target.items()):
   props[field]={'type':'string','enum':['0','1','2','3','4'],'description':RUBRICS[field]+' Select the ordinal level as a string; levels increase in the stated attribute.'};gold[field]=str(value)
  traces.append({'operation':'ordinal_ratings_to_enums','fields':sorted(target),'note':'Native ordered ratings, not confidence probabilities; rubric dimensions preserved. Anchors paraphrase HelpSteer2 Appendix G.3.1.','rubric_source':RUBRIC_SOURCE})
 elif source in ['MultiRC','Aegis2','CivilComments','RuleTaker']:
  props['decision']={'type':'boolean','description':q}
  if source=='MultiRC': state={'passage':state,'candidate_answer':options[0]};q=q+' Is the candidate answer correct?';props['decision']['description']=q;gold['decision']=target;traces.append({'operation':'native_candidate_correctness','independent_answers':True})
  elif source=='RuleTaker':
   assert target in {'entailment','not entailment'}, (rid,'invalid native entailment label',target)
   q='Does the hypothesis follow from the provided facts and rules? Hypothesis: '+q;props['decision']['description']=q;gold['decision']=target=='entailment';traces.append({'operation':'native_binary_entailment','positive_label':'entailment','negative_label':'not entailment','note':'Corrects unsupported three-way staging options; does not distinguish contradiction and unknown.'})
  elif source=='CivilComments':
   gold['decision']=target>=0.5; q='Would at least half of annotators consider this comment toxic: rude, disrespectful, or unreasonable language likely to make someone leave the discussion?' 
   props['decision']['description']=q;metadata['annotation_fractions']={k:v for k,v in r['raw_native'].items() if k!='text'};traces.append({'operation':'annotation_fraction_threshold','attribute':'toxicity','threshold':0.5,'comparison':'>=','note':'Hard recast of observed annotation fraction; neither model confidence nor objective truth.'})
  else:gold['decision']=target;traces.append({'operation':'native_human_safety_boolean'})
 elif source=='GSM8K':
  native=target['numeric_answer'];num=decimal.Decimal(native.replace(',','')); assert num.is_finite()
  def fmt(d): return format(d,'f').rstrip('0').rstrip('.') if '.' in format(d,'f') else format(d,'f')
  with decimal.localcontext() as context:
   context.prec=max(50,len(num.as_tuple().digits)+abs(num.as_tuple().exponent)+10);opts=[fmt(num+decimal.Decimal(i)) for i in [0,-2,-1,1,2]]
  assert len(set(opts))==5,(rid,'non-distinct numeric alternatives')
  props['decision'],gold['decision'],trace=choice(opts,0,seed,rid);traces.extend([{'operation':'arithmetic_distractors','offsets':[0,-2,-1,1,2],'native_numeric_answer':native,'note':'Synthetic numeric alternatives; not independently sampled human errors.'},trace])
 else:
  if source=='HelpSteer2':metadata['preference_strength']=target['preference_strength'];target=target['choice']
  props['decision'],gold['decision'],trace=choice(options,target,seed,rid);traces.append(trace)
 schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
 record={'id':rid,'source':source,'family_id':r.get('family_id',r.get('group_id')),'model_payload':{'input':canonical({'state':state,'question':q}),'schema':schema},'gold':gold,'field_types':{k:('noul' if p['type']=='boolean' else 'score' if source=='HelpSteer2' and r['source_config']=='ratings' else 'choice') for k,p in props.items()},'target_metadata':metadata,'transformation_trace':traces,'generator':{'version':1,'seed':seed,'seed_derivation':'SHA256(UTF8(canonical JSON [seed,record_id,field,original_option_index])); ascending digest order; no Python hash'},'native_source':{k:r[k] for k in ['source_split','split','source_config','source_revision','native_revision','source_index','source_row_index','provenance'] if k in r}}
 validate(record);return record

def validate(record):
 schema=record['model_payload']['schema'];gold=record['gold'];assert set(gold)==set(schema['properties'])
 assert 'raw_native' not in record['model_payload'] and 'gold' not in record['model_payload']
 for key,s in schema['properties'].items():
  if s['type']=='boolean':assert type(gold[key]) is bool,(record['id'],gold)
  else:assert type(gold[key]) is str and gold[key] in s['enum'],(record['id'],gold)

def generate(input_path,output_path,manifest_path,seed=SEED,expected=EXPECTED_INPUT):
 input_hash=sha(input_path)
 if expected and input_hash!=expected:raise ValueError(f'Input checksum mismatch: {input_hash} != {expected}')
 rows=[json.loads(line) for line in input_path.read_text().splitlines()];assert len(rows)==5000 and len({r['id'] for r in rows})==5000
 typed=[convert(r,seed) for r in rows]; output_path.parent.mkdir(parents=True,exist_ok=True);output_path.write_bytes((''.join(canonical(r)+'\n' for r in typed)).encode())
 manifest={'generator_version':1,'seed':seed,'input_path':str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),'input_sha256':input_hash,'output_sha256':sha(output_path),'record_count':len(typed),'source_counts':dict(sorted(collections.Counter(r['source'] for r in typed).items())),'field_type_counts':dict(collections.Counter(p['type'] for r in typed for p in r['model_payload']['schema']['properties'].values())),'code_sha256':sha(Path(__file__)),'canonical_serialization':'UTF-8; sorted object keys; compact separators; ensure_ascii=False; allow_nan=False; LF after every record; preserve input row order','permutation':'SHA256 of canonical JSON [seed, record_id, field, original_option_index], sorted ascending by digest with index tie break','no_model_calls':True,'notes':['One typed record per native record; HelpSteer2 ratings retain five separate fields.','CivilComments uses toxicity annotation fraction >=0.5; native annotation fractions retained only outside model payload.','GSM8K synthetic alternatives are exact Decimal answer plus offsets -2,-1,+1,+2.','No calibration probabilities invented. Scores use enum strings for Nimble flat boolean/enum schema compatibility.','Original labels, rationales and raw_native are absent from model payload; gold and transformation trace are audit-only.']}
 manifest_path.parent.mkdir(parents=True,exist_ok=True);manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return manifest
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--input',type=Path,default=ROOT/'data/training/sample-5000.jsonl');a.add_argument('--output',type=Path,default=ROOT/'data/training/typed-5000.jsonl');a.add_argument('--manifest',type=Path,default=ROOT/'data/training/manifests/typed-5000.json');a.add_argument('--seed',type=int,default=SEED);args=a.parse_args();print(json.dumps(generate(args.input,args.output,args.manifest,args.seed),indent=2))
