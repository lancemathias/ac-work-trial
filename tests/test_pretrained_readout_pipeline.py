"""Exercise padding, typed scoring, gold exclusion and calibration on CPU."""
import copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from torch import nn
from pretrained_readout import predict
from evaluate_pretrained_readout import prepare_decisions
from evaluate_trained_joint import answer_body
from run_baselines import score,record,metrics as benchmark_metrics
from train_joint import fit_temperature,metrics
from test_pretrained_readout import Tokenizer

class Text(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=nn.Embedding(16,4)
 def forward(self,input_ids,attention_mask,**kwargs):
  return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.model=Text();self.output=nn.Linear(4,16,bias=False)
 def get_output_embeddings(self):return self.output

class PipelineTests(unittest.TestCase):
 def test_batched_last_valid_token_scoring_and_calibrated_benchmark_contract(self):
  torch.manual_seed(7);model=Model().requires_grad_(False).eval()
  prepared=[{'id':'short','type':'choice','target_index':0,'source':'fixture','encoding':{'input_ids':[1,2,3],'option_ids':['a','z'],'code_token_ids':[1,2]}},
   {'id':'long','type':'choice','target_index':1,'source':'fixture','encoding':{'input_ids':[4,5,6,7,8],'option_ids':['a','z'],'code_token_ids':[1,2]}}]
  batched=predict(model,Tokenizer(),prepared,batch_size=2);single=predict(model,Tokenizer(),prepared,batch_size=1)
  for a,b in zip(batched,single):torch.testing.assert_close(torch.tensor(a['logits']),torch.tensor(b['logits']))
  fitted=fit_temperature(batched);self.assertLessEqual(fitted['calibrated_nll'],fitted['raw_nll']+1e-12)
  self.assertEqual(metrics(batched)['accuracy'],metrics(batched,fitted['temperature'])['accuracy'])
  scored=[]
  for item in batched:
   row={'id':item['id'],'suite':'fixture','group':item['id'],'source':{},'state':'public','question':{'type':'choice','instructions':'Choose','criteria':{'a':'first','z':'last'}},'label':['a','z'][item['target_index']]}
   probs=(torch.tensor(item['logits'])/fitted['temperature']).softmax(-1).tolist()
   body=answer_body(row['question'],item['option_ids'],probs)
   scored.append(record(row,dict(score(row,body),response=body,latency_ms=item['latency_ms'])))
  summary=benchmark_metrics(scored,2);self.assertEqual(summary['valid'],2);self.assertGreater(summary['median_latency_ms'],0)
  self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
 def test_benchmark_gold_and_provenance_cannot_enter_prepared_inputs(self):
  row={'id':'x','state':'public','question':{'type':'noul','instructions':'Is it correct?'},'label':True,'source':{'audit':'secret'}}
  a=prepare_decisions(Tokenizer(),[row]);changed=copy.deepcopy(row);changed['label']=False;changed['source']['audit']='different'
  self.assertEqual(a,prepare_decisions(Tokenizer(),[changed]));self.assertEqual(set(a[0]),{'id','type','encoding'})
  adapted={'id':'x','source':'allowed-training-task','request':{'state':row['state'],'questions':{'q':row['question']}},'targets':{'q':'true'}}
  cal=prepare_decisions(Tokenizer(),[adapted],True);self.assertEqual(cal[0]['target_index'],0);self.assertEqual(cal[0]['encoding'],a[0]['encoding'])
if __name__=='__main__':unittest.main()
