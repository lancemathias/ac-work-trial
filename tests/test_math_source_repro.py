import copy,json,tempfile,unittest
from pathlib import Path
from math_source_repro import compare,runtime_config

class MathSourceReproTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.a=Path(self.tmp.name)/'a';self.b=Path(self.tmp.name)/'b'
  for root in [self.a,self.b]:
   (root/'training').mkdir(parents=True);(root/'test').mkdir()
   for name in ['initial-head.safetensors','best-head.safetensors','best.json','temperature.json','dev-traces.jsonl','calibration-traces.jsonl']:(root/'training'/name).write_text('identical')
   rows=[dict(event='train_step',step=i+1,epoch=i//102+1,loss=.5,learning_rate=.001,gradient_norm=.2,decisions=64,record_ids=[str(i)],timestamp_unix=1) for i in range(510)]
   (root/'training/metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
   for name in ['raw-predictions.jsonl','calibrated-predictions.jsonl']:
    (root/'test'/name).write_text(''.join(json.dumps(dict(id=str(i),probabilities=[.4,.6],latency_ms=1))+'\n' for i in range(2111)))
 def test_exact_full_repeats_pass_and_latency_is_ignored(self):
  p=self.b/'test/calibrated-predictions.jsonl';p.write_text(p.read_text().replace('"latency_ms": 1','"latency_ms": 50'))
  self.assertEqual(compare(self.a,self.b)['status'],'passed')
 def test_one_loss_change_blocks_gate(self):
  p=self.b/'training/metrics.jsonl';p.write_text(p.read_text().replace('"loss": 0.5','"loss": 0.50000001',1))
  self.assertFalse(compare(self.a,self.b)['checks']['all_training_metrics'])
 def test_matching_short_training_does_not_pass(self):
  for root in [self.a,self.b]:
   p=root/'training/metrics.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
  self.assertEqual(compare(self.a,self.b)['status'],'failed')
 def test_matching_incomplete_predictions_do_not_pass(self):
  for root in [self.a,self.b]:
   p=root/'test/calibrated-predictions.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
  self.assertEqual(compare(self.a,self.b)['status'],'failed')
 def test_one_probability_change_blocks_gate(self):
  p=self.b/'test/raw-predictions.jsonl';p.write_text(p.read_text().replace('[0.4, 0.6]','[0.40000001, 0.59999999]',1))
  self.assertFalse(compare(self.a,self.b)['checks']['raw-predictions.jsonl'])
 def test_runtime_only_changes_backend_and_run_name(self):
  source={'selection_metric':'accuracy','run_name':'old','dataset':{'sha256':'fixed'},'seed':7,'head':{'width':512}}
  before=copy.deepcopy(source);result=runtime_config(source,'new')
  self.assertEqual(source,before);result.pop('head_sdpa_backend');result['run_name']='old';self.assertEqual(result,source)
  with self.assertRaises(ValueError):runtime_config({'selection_metric':'nll'},'bad')
