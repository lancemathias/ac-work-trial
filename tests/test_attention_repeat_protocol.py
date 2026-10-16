import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from attention_repeat_protocol import validate_configs,movement_probe

class RepeatProtocolTests(unittest.TestCase):
 def fixture(self,root):
  base={'seed':20261014,'selection_metric':'accuracy','learning_rate':.0003};rows=[]
  for seed in [20261015,20261016]:
   for mode in ['native_causal','bidirectional_full_attention']:
    name=f'{seed}-{mode}.json';cfg=dict(base,seed=seed,split_seed=20261014,attention_mode=mode);raw=json.dumps(cfg).encode();(root/name).write_bytes(raw);rows.append({'config':name,'sha256':hashlib.sha256(raw).hexdigest()})
  return base,{'runs':rows}
 def test_training_seeds_vary_without_resplitting(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);base,receipt=self.fixture(root);cfgs=validate_configs(root,receipt,base)
   self.assertEqual({c['split_seed'] for c in cfgs},{20261014});self.assertEqual({c['seed'] for c in cfgs},{20261015,20261016})
 def test_rejects_coupled_split_seed_even_with_valid_checksum(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);base,receipt=self.fixture(root);row=receipt['runs'][0];p=root/row['config'];cfg=json.loads(p.read_text());cfg['split_seed']=cfg['seed'];raw=json.dumps(cfg).encode();p.write_bytes(raw);row['sha256']=hashlib.sha256(raw).hexdigest()
   with self.assertRaisesRegex(ValueError,'separate'):validate_configs(root,receipt,base)
 def test_rejects_optimizer_change_even_with_valid_checksum(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);base,receipt=self.fixture(root);row=receipt['runs'][0];p=root/row['config'];cfg=json.loads(p.read_text());cfg['learning_rate']=.001;raw=json.dumps(cfg).encode();p.write_bytes(raw);row['sha256']=hashlib.sha256(raw).hexdigest()
   with self.assertRaisesRegex(ValueError,'recipe'):validate_configs(root,receipt,base)
 def test_movement_gate_checks_actual_output_and_frozen_state(self):
  import torch
  from types import SimpleNamespace
  class Text(torch.nn.Module):
   def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(8,4)
   def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.model=Text()
  m=Model().requires_grad_(False).eval();before={k:v.clone() for k,v in m.state_dict().items()};result=movement_probe(m,'cpu','cpu');self.assertTrue(result['exact'])
  for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
  m.requires_grad_(True)
  with self.assertRaisesRegex(ValueError,'frozen'):movement_probe(m,'cpu','cpu')
if __name__=='__main__':unittest.main()
