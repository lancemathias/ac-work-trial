"""CPU-only native thinking-template feasibility; no generation or scoring."""
import hashlib,json,sys
from pathlib import Path
import modal
if Path('/project/scripts').is_dir():sys.path.insert(0,'/project/scripts')
from modal_pretrained_readout import ROOT,ENV,image,models,results
app=modal.App('ac-worktrial-thinking-readout-preflight')
@app.function(image=image,cpu=4,memory=16384,timeout=900,retries=0,scaledown_window=2,volumes={'/models':models.read_only(),'/results':results.read_only()})
def inspect():
 from transformers import AutoProcessor
 from pretrained_readout import messages,encode
 from evaluate_pretrained_readout import prepare_decisions
 from training_data import load_splits,digest
 from run_baselines import load_panel
 cfg=json.loads(Path('/project/configs/e4-qwen38-27b-v3.json').read_text());tok=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 open_ids=tok('<think>',add_special_tokens=False).input_ids;close_ids=tok('</think>',add_special_tokens=False).input_ids
 if len(open_ids)!=1 or len(close_ids)!=1:return {'status':'infeasible','reason':'Native thinking tags must be single tokens','open_ids':open_ids,'close_ids':close_ids}
 splits,_=load_splits('/results/e3-medium-v3-seed20261014-e7dd4bb-splits/manifest.json');reference=json.loads(Path('/project/readout-preflight.json').read_text());sets={};suffixes=set();mode_headers=set()
 for role in ['calibration','dev','test']:
  rows=load_panel() if role=='test' else splits[role];native=prepare_decisions(tok,rows,role!='test');native_fp=hashlib.sha256();thinking_fp=hashlib.sha256();lengths=[];distinct=0;overlength=[]
  for item,row in zip(native,rows):
   native_fp.update(json.dumps({'id':item['id'],**item['encoding']},sort_keys=True,separators=(',',':')).encode())
   request=row['request'] if role!='test' else {'state':row['state'],'questions':{'q':row['question']}};turns,option_ids=messages(request['state'],request['questions']['q'])
   base=tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=False,enable_thinking=False)
   thinking_base=tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=False,enable_thinking=True)
   text=tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=True,enable_thinking=True)
   native_text=tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   if not text.startswith(thinking_base) or not native_text.startswith(base):raise ValueError('Generation prefix changes mode-specific serialized source messages')
   provided=turns[0]['content'];a=base.index(provided);b=thinking_base.index(provided)
   if thinking_base[:a]!=base[:a] or b<a or thinking_base[b:]!=base[a:]:raise ValueError('Thinking changes supplied system/user content')
   mode_headers.add(thinking_base[a:b])
   suffix=text[len(thinking_base):];native_suffix=native_text[len(base):];suffixes.add((native_suffix,suffix))
   if not suffix.rstrip().endswith('<think>') or '</think>' in suffix:raise ValueError('No open native thinking segment')
   ids=tok(text,add_special_tokens=False).input_ids
   if not ids:raise ValueError('Empty thinking prompt')
   if len(ids)>8192:overlength.append({'id':item['id'],'tokens':len(ids)})
   distinct+=ids!=item['encoding']['input_ids'];lengths.append(len(ids))
   e={'input_ids':ids,'code_token_ids':item['encoding']['code_token_ids'],'option_ids':option_ids};thinking_fp.update(json.dumps({'id':item['id'],**e},sort_keys=True,separators=(',',':')).encode())
  if native_fp.hexdigest()!=reference['sets'][role]['encodings_sha256'] or len(native)!=reference['sets'][role]['expected']:raise ValueError('Native baseline input changed')
  sets[role]={'records':len(rows),'native_encodings_sha256':native_fp.hexdigest(),'thinking_encodings_sha256':thinking_fp.hexdigest(),'distinct_thinking_inputs':distinct,'overlength':overlength,'maximum_prompt_tokens':max(lengths),'total_prompt_tokens':sum(lengths),'reasoning_token_budget':128,'max_final_tokens_upper_bound':max(lengths)+128+8}
 if len(mode_headers)!=1 or len(suffixes)!=1 or any(s['distinct_thinking_inputs']!=s['records'] for s in sets.values()):raise ValueError('Thinking template suffix must be distinct and shared across all inputs')
 native_suffix,thinking_suffix=next(iter(suffixes))
 return {'status':'infeasible_as_written' if any(s['overlength'] for s in sets.values()) else 'passed','native_thinking_mode_system_header':next(iter(mode_headers)),'backbone_id':cfg['backbone_id'],'revision':cfg['backbone_revision'],'open_think_token_id':open_ids[0],'close_think_token_id':close_ids[0],'native_generation_suffix':native_suffix,'thinking_generation_suffix':thinking_suffix,'sets':sets,'scope':'CPU template/tokenizer feasibility only; no generated text, parameter fitting, GPU inference or benchmark scoring','source_sha256':{n:digest(Path('/project')/n) for n in ['scripts/modal_thinking_readout_preflight.py','scripts/pretrained_readout.py','scripts/evaluate_pretrained_readout.py']}}
@app.local_entrypoint()
def main():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 out=ROOT/'outputs/autoresearch/thinking-readout-preflight.json'
 if out.exists():raise FileExistsError('Preserve prior receipt')
 value=inspect.remote();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))

@app.function(image=image,cpu=2,memory=8192,timeout=180,retries=0,scaledown_window=2,volumes={'/models':models.read_only()})
def template_diagnostic():
 from transformers import AutoProcessor
 from pretrained_readout import messages
 cfg=json.loads(Path('/project/configs/e4-qwen38-27b-v3.json').read_text());tok=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',local_files_only=True).tokenizer
 turns,_=messages('PUBLIC_STATE_SENTINEL',{'type':'choice','instructions':'PUBLIC_QUESTION_SENTINEL','criteria':{'a':'PUBLIC_OPTION_A','b':'PUBLIC_OPTION_B'}})
 out={}
 for mode in [False,True]:
  base=tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=False,enable_thinking=mode);full=tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=True,enable_thinking=mode)
  out[str(mode)]={'base':base,'full':full,'base_is_prefix':full.startswith(base),'generation_suffix':full[len(base):] if full.startswith(base) else None,'source_sentinels_preserved':all(x in full for x in ['PUBLIC_STATE_SENTINEL','PUBLIC_QUESTION_SENTINEL','PUBLIC_OPTION_A','PUBLIC_OPTION_B'])}
 return {'status':'diagnosed','modes':out,'source':'Only public sentinel fixture; no benchmark task or generated answer','tokenizer_template_sha256':hashlib.sha256(str(tok.chat_template).encode()).hexdigest()}
@app.local_entrypoint()
def diagnose():
 from modal.config import _profile,config
 assert _profile=='ac-worktrial' and config.get('environment')==ENV
 result=template_diagnostic.remote();(ROOT/'outputs/autoresearch/thinking-template-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
