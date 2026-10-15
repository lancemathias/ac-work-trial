"""Exercise the actual evaluator's reject/pass branches without GPU or W&B I/O."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import torch
from test_pretrained_readout import Tokenizer
from evaluate_pretrained_readout import prepare_decisions
from evaluate_bounded_thinking import _evaluate_once
from run_baselines import ROOT

class ThinkingRuntimeTests(unittest.TestCase):
 def check_runtime(self,advance):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);baseline=root/'baseline';baseline.mkdir();out=root/'output'
   cfg=json.loads((ROOT/'configs/e4-qwen38-27b-v3.json').read_text());tokenizer=Tokenizer()
   splits={role:[{'id':role+str(i),'source':'allowed','request':{'state':'public','questions':{'q':{'type':'noul','instructions':'True?'}}},'targets':{'q':['true','false'][i]}} for i in range(2)] for role in ['calibration','dev']}
   panel=[{'id':'test'+str(i),'suite':'fixture','group':'test'+str(i),'source':{},'state':'public','question':{'type':'noul','instructions':'True?'},'label':i==0} for i in range(2)]
   checks={}
   def write(name,value,lines=False):
    text=''.join(json.dumps(x)+'\n' for x in value) if lines else json.dumps(value)
    (baseline/name).write_text(text);checks[name]=hashlib.sha256(text.encode()).hexdigest()
   write('config.json',dict(backbone_id=cfg['backbone_id'],backbone_revision=cfg['backbone_revision'],checkpoint_sha256='baseline',panel_sha256='fixture'))
   write('completion.json',{'status':'complete','parameters_unchanged':True})
   preflight={'sets':{}}
   for role in ['calibration','dev','test']:
    native=[]
    for i in range(2):
     logits=([0.,.1] if advance and role in ['dev','test'] and i==0 else ([2.,0.] if i==0 else [0.,2.]))
     r={'id':role+str(i),'type':'noul','option_ids':['true','false'],'logits':logits,'latency_ms':1,'input_tokens':3}
     if role!='test':r.update(target_index=i,source='allowed')
     native.append(r)
    if role!='test' or advance:write(role+'-logits.jsonl',native,True)
    prepared=prepare_decisions(tokenizer,panel if role=='test' else splits[role],role!='test',option_order='reverse');h=hashlib.sha256()
    for x in prepared:h.update(json.dumps({'id':x['id'],**x['encoding']},sort_keys=True,separators=(',',':')).encode())
    preflight['sets'][role]={'records':2,'encodings_sha256':h.hexdigest()}
   (baseline/'checksums.json').write_text(json.dumps(checks))
   calls=[]
   def predict(model,tokenizer,prepared,settings,emit):
    result=[];calls.append(prepared[0]['id'])
    for i,x in enumerate(prepared):
     item={k:v for k,v in x.items() if k!='encoding'};item.update(option_ids=x['encoding']['option_ids'],logits=([0.,2.] if i==0 else [2.,0.]),latency_ms=1,input_tokens=3,expanded_tokens=4);result.append(item);emit(item)
    return result
   model=torch.nn.Linear(2,2).requires_grad_(False).eval();wb=MagicMock(id='cpu-fixture',url='fixture')
   with patch('training_data.load_splits',return_value=(splits,{})),patch('resident_backbone.get_processor',return_value=SimpleNamespace(tokenizer=tokenizer)),patch('resident_backbone.get_model',return_value=model),patch('kernel_runtime.configure_optimized_runtime',return_value={'mode':'cpu_fixture'}),patch('bounded_thinking_readout.predict',side_effect=predict),patch('bounded_thinking_readout.prepare',side_effect=lambda tok,rows,settings,training_sources=False:prepare_decisions(tok,rows,training_sources,option_order='reverse')),patch('wandb.init',return_value=wb),patch('run_baselines.load_panel',return_value=panel) as load_panel,patch('report_trained_joint.report',return_value={'macro_coverage_adjusted_accuracy':1.,'suites':{}}):
    result=_evaluate_once(out,'fixture',10**12,lambda:None,baseline,preflight,'fixture')
   completion=json.loads((out/'completion.json').read_text())
   if advance:
    self.assertEqual(result['status'],'complete');load_panel.assert_called_once();self.assertEqual(calls,['calibration0','dev0','test0']);self.assertEqual(completion['attempted'],2)
    scored=[json.loads(x) for x in (out/'calibrated-predictions.jsonl').read_text().splitlines()]
    self.assertTrue(all(x['correct'] for x in scored));self.assertTrue(all(abs(sum(x['probabilities'])-1)<1e-10 for x in scored))
   else:
    self.assertEqual(result['status'],'rejected_on_dev');load_panel.assert_not_called();self.assertEqual(calls,['calibration0','dev0']);self.assertFalse(completion['test_read']);self.assertFalse(completion['test_inference']);self.assertFalse(completion['test_scored']);self.assertFalse((out/'calibrated-predictions.jsonl').exists())
   self.assertTrue(completion['parameters_unchanged']);self.assertTrue((out/'checksums.json').exists())
 def test_rejected_dev_never_reads_test_logits_or_infers_scores_benchmarks(self):self.check_runtime(False)
 def test_passed_dev_scores_calibrated_semantic_probabilities(self):self.check_runtime(True)
