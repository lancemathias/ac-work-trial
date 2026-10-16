"""Same-record closed-world instruction clarification; never relabel native data."""
import copy,hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD='Does the hypothesis follow from the provided facts and rules? Hypothesis: '
NEW=('Use closed-world reasoning for this rule task. Apply the facts and rules by forward chaining. '
 'Treat a positive fact that cannot be derived as false, and its negation as true. '
 'Negated rule conditions use negation as failure. Is the following hypothesis true under these rules? Hypothesis: ')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def prepare(source,destination,native_path,config_path):
 source,destination,native_path,config_path=map(Path,[source,destination,native_path,config_path])
 if destination.exists() or config_path.exists():raise FileExistsError('Preserve prior data/config')
 release=json.loads((ROOT/'outputs/autoresearch/source-addback-release.json').read_text())
 if sha(native_path)!=release['v1_native_sha256']:raise ValueError('Native snapshot changed')
 native={r['id']:r for r in map(json.loads,native_path.read_text().splitlines())};additions=[json.loads(x) for x in (source/'additions.jsonl').read_text().splitlines()]
 if len(additions)!=250 or {r['suite'] for r in additions}!={'RuleTaker'}:raise ValueError('Expected original250RuleTaker records')
 cases=[];synthetic=[]
 for row in additions:
  raw=native[row['id']]['raw_native']
  if row['state']!=raw['context'] or row['question']['instructions']!=OLD+raw['question'] or row['label']!=(raw['label']=='entailment'):raise ValueError('Native payload/label mapping differs')
  if re.search(r'\bnot\b',raw['question'].lower()) and not re.search(r"\b(?:not|no|never|neither|nor)\b|n['’]t",raw['context'].lower()) and row['label'] is True:
   item={'id':row['id'],'config':raw['config'],'native_label':raw['label']};cases.append(item)
   if 'NatLang' not in raw['config']:synthetic.append(item)
 destination.mkdir(parents=True);counts={};changed_ids={r['id'] for r in additions}
 for name in ['dataset.jsonl','train.jsonl','additions.jsonl']:
  count=0;lines=[]
  for line in (source/name).read_text().splitlines(keepends=True):
   row=json.loads(line)
   if row['id'] in changed_ids:
    before=copy.deepcopy(row)
    if not row['question']['instructions'].startswith(OLD):raise ValueError('Unexpected original instruction')
    row['question']['instructions']=NEW+row['question']['instructions'][len(OLD):];restored=copy.deepcopy(row);restored['question']['instructions']=before['question']['instructions']
    if restored!=before:raise ValueError('Unintended data change')
    line=canonical(row)+'\n';count+=1
   lines.append(line)
  if count!=250:raise ValueError('Exactly250requests must change in each artifact')
  (destination/name).write_text(''.join(lines));counts[name]=count
 for name in ['dev.jsonl','calibration.jsonl']:shutil.copy2(source/name,destination/name)
 shutil.copy2(source/'hygiene-report.json',destination/'parent-hygiene-report.json');hygiene=json.loads((source/'hygiene-report.json').read_text());hygiene.update(dataset_sha256=sha(destination/'dataset.jsonl'),parent_dataset_sha256=sha(source/'dataset.jsonl'),basis='Same previously screened source IDs/groups/states/labels; only250RuleTaker instructions clarified. No new rows, contexts or benchmark-source tasks. Original complete benchmark pool exclusion proof inherited.',parent_report_sha256=sha(destination/'parent-hygiene-report.json'))
 (destination/'hygiene-report.json').write_text(json.dumps(hygiene,indent=2)+'\n');manifest=json.loads((source/'manifest.json').read_text());manifest['source_dataset_sha256']=sha(destination/'dataset.jsonl');manifest['hygiene']['report']['sha256']=sha(destination/'hygiene-report.json');manifest['splits']['train']['points']['sha256']=sha(destination/'train.jsonl');manifest['split_rule']='Exact original source condition rows/order/labels; only250RuleTaker instruction prefixes clarified'
 (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');cfg=json.loads((ROOT/'configs/source-addbacks/ruletaker.json').read_text());cfg['dataset'].update(path='/project/ruletaker-cwa-explicit/dataset.jsonl',sha256=sha(destination/'dataset.jsonl'),version='same-record-ruletaker-explicit-closed-world');cfg.update(head_sdpa_backend='math',run_name='ruletaker-cwa-explicit-seed20261014',purpose='Same250RuleTaker records/native labels with explicit closed-world/NAF instructions. Compare originalRule addback and existing-source control; cardruletaker-explicit-closed-world. Same6512/809/857,510updates, devaccuracy selection and separateT.');config_path.parent.mkdir(parents=True,exist_ok=True);config_path.write_text(json.dumps(cfg,indent=2)+'\n')
 proof={'status':'prepared_not_GPU_queued','changed_counts':counts,'native_rows_verified':250,'native_snapshot_sha256':sha(native_path),'native_labels_unchanged':True,'states_hypotheses_groups_ids_row_order_unchanged':True,'heldouts_byte_identical':all(sha(source/n)==sha(destination/n) for n in ['dev.jsonl','calibration.jsonl']),'lexical_CWA_cases':len(cases),'unambiguous_synthetic_positive_Horn_cases':len(synthetic),'synthetic_cases':synthetic,'NatLang_cases_need_separate_linguistic_proof':len(cases)-len(synthetic),'original_instruction_prefix':OLD,'clarified_instruction_prefix':NEW,'parent_manifest_sha256':sha(source/'manifest.json'),'files':{p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()},'config_sha256':sha(config_path),'parent_hygiene_inherited_without_new_source_data':True}
 (destination/'release.json').write_text(json.dumps(proof,indent=2)+'\n');return proof
if __name__=='__main__':
 result=prepare(ROOT/'data/training/source-addbacks-v2/ruletaker',ROOT/'data/training/ruletaker-cwa-explicit',ROOT/'data/training/sample-5000.jsonl',ROOT/'configs/source-quality/ruletaker-cwa-explicit.json');(ROOT/'outputs/autoresearch/ruletaker-cwa-release.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
