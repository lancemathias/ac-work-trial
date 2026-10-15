"""Exercise repair orchestration without model loading, extraction or network."""
import json,tempfile,hashlib,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock,patch
import run_ruletaker_cwa as runner
from head_probe_io import write_json
from training_data import digest
from run_baselines import ROOT

class RepairRunner(unittest.TestCase):
 def exercise(self,interrupt=False,tamper=False):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);project=root/'project';data=root/'data';project.mkdir();data.mkdir();(project/'configs/source-quality').mkdir(parents=True);dataset=project/'ruletaker-cwa-explicit';dataset.mkdir()
   cfg=json.loads((ROOT/'configs/source-quality/ruletaker-cwa-explicit.json').read_text());cfgpath=project/'configs/source-quality/ruletaker-cwa-explicit.json';write_json(cfgpath,cfg)
   write_json(project/'configs/source-quality/ruletaker-cwa-cache-source.json',{})
   (dataset/'dataset.jsonl').write_text('fixture\n');(dataset/'additions.jsonl').write_text('changed fixture\n')
   write_json(project/'ruletaker-cwa-cpu.json',{'status':'passed','source_sha256':{'configs/source-quality/ruletaker-cwa-explicit.json':digest(cfgpath)},'dataset_sha256':digest(dataset/'dataset.jsonl')})
   if tamper:(dataset/'dataset.jsonl').write_text('tampered\n')
   base=data/'hidden-cache/qwen35-9b-v3-fast-v1';base.mkdir(parents=True);write_json(base/'manifest.json',{'fixture':'unchanged base'})
   (project/'scripts').mkdir();(project/'scripts/benchmark_training_data.py').write_text('fixture')
   def mapped(path):
    p=Path(path)
    for prefix,dest in [('/project',project),('/data',data)]:
     if str(p)==prefix or str(p).startswith(prefix+'/'):return dest/str(p)[len(prefix):].lstrip('/')
    return p
   def extract(snapshot,source,schema,destination,checkpoint):
    destination.mkdir(exist_ok=True);write_json(destination/'manifest.json',{'fixture':'new250'});checkpoint()
   def compose(parts,destination,rows,config,allow_source_subset):
    destination.mkdir();m={'status':'complete','identity':{'parent_manifest_sha256':[p['manifest_sha256'] for p in parts],'training_source_config':config,'record_count':8178}};write_json(destination/'manifest.json',m);return m
   calls=[]
   def train(runtime,manifest,output,**kwargs):
    calls.append(output);output.mkdir();self.assertEqual(kwargs['expected_initial_head_sha256'],'909b3c9ff1f565479d2bd81a1346564af2f5f0a433f2bf869fa9e7cad4ed573e')
    self.assertEqual(json.loads(runtime.read_text())['selection_metric'],'accuracy')
    if interrupt and len(calls)==1:raise RuntimeError('fixture interruption')
    write_json(output/'summary.json',{'steps':510});write_json(output/'wandb-run.json',{'id':'fixture'})
   def evaluate(training,test,commit,callback):test.mkdir();(test/'calibrated-predictions.jsonl').write_text('fixture\n');callback()
   output=root/'run';checkpoint=Mock();data_checkpoint=Mock()
   with ExitStack() as stack:
    stack.enter_context(patch.object(runner,'Path',side_effect=mapped));stack.enter_context(patch('run_baselines.ROOT',project))
    extract_mock=stack.enter_context(patch('cache_hidden_runtime_batch4.extract',side_effect=extract));stack.enter_context(patch('compose_feature_cache.compose',side_effect=compose));stack.enter_context(patch('training_data.load_splits',return_value=({'train':[{'id':'fixture'}]},{})))
    stack.enter_context(patch('train_joint.run',side_effect=train));stack.enter_context(patch('evaluate_trained_joint.evaluate',side_effect=evaluate));stack.enter_context(patch('report_trained_joint.report',return_value={'macro_coverage_adjusted_accuracy':.7,'suites':{}}));stack.enter_context(patch('report_prediction_dominance.summarize',return_value={}));stack.enter_context(patch('wandb.Api',side_effect=RuntimeError('fixture network unavailable')));stack.enter_context(patch('torch.cuda.empty_cache'))
    if tamper:
     with self.assertRaisesRegex(ValueError,'records changed'):runner.run(output,'fixture',{},checkpoint,data_checkpoint,10**12)
     extract_mock.assert_not_called();self.assertFalse(output.exists());return
    if interrupt:
     with self.assertRaisesRegex(RuntimeError,'fixture interruption'):runner.run(output,'fixture',{},checkpoint,data_checkpoint,10**12)
    result=runner.run(output,'fixture',{},checkpoint,data_checkpoint,10**12)
    self.assertEqual(result['status'],'complete');self.assertEqual(len(calls),2 if interrupt else 1)
    attempt=output/result['attempt'];self.assertTrue((attempt/'test/wandb-update-warning.json').exists());self.assertTrue((output/'checksums.json').exists())
    count=extract_mock.call_count;self.assertEqual(runner.run(output,'fixture',{},checkpoint,data_checkpoint,10**12),result);self.assertEqual(extract_mock.call_count,count)
    self.assertEqual(len(list((output/'attempts').iterdir())),2 if interrupt else 1)
 def test_complete_sealed_and_recovered_without_new_work(self):self.exercise()
 def test_interrupted_attempt_preserved_and_completed_attempt_recovered(self):self.exercise(interrupt=True)
 def test_changed_records_block_before_any_extraction(self):self.exercise(tamper=True)
