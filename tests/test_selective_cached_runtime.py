"""Exercise complete cached scoring with synthetic data; real benchmark files absent."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import apply_selective_cached_test as module

class CachedRuntimeTests(unittest.TestCase):
 def exercise(self,reject=None):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);results=root/'results';results.mkdir();native=results/'readout-fast-qwen38-27b-v1';reverse=results/'readout-native-reverse-pool-v1';thinking=results/'readout-bounded-thinking128-staged-v2/attempts/abc123';dev=root/'development';output=root/'candidate'
   def write(directory,files):
    directory.mkdir(parents=True,exist_ok=True);checks={}
    for name,value in files.items():
     raw=(''.join(json.dumps(x)+'\n' for x in value) if name.endswith('.jsonl') else json.dumps(value)).encode();(directory/name).write_bytes(raw);checks[name]=hashlib.sha256(raw).hexdigest()
    (directory/'checksums.json').write_text(json.dumps(checks))
   a=[dict(id=str(i),type='noul',option_ids=['true','false'],logits=[2.,0.],latency_ms=1,input_tokens=10) for i in range(2111)]
   b=[dict(r,option_ids=['false','true'],logits=[3.,0.] if r['id']=='0' else [0.,2.]) for r in a]
   c=[dict(r,logits=[4.,0.],expanded_tokens=20) for r in a]
   write(native,{'config.json':{'checkpoint_sha256':'frozen','panel_sha256':'fixed'},'test-logits.jsonl':a});write(reverse,{'reverse-test-logits.jsonl':b})
   write(thinking,{'config.json':{'parent_checkpoint_sha256':'frozen','panel_sha256':'fixed'},'completion.json':{'status':'complete','attempted':2111,'parameters_unchanged':reject!='parameters','optimizer_updates':0},'test-logits.jsonl':c})
   fit={'status':'dev_gate_passed','selection_metric':'accuracy','gate':{'status':'passed','pooled_dev_accuracy':.8430160692212608,'selective_dev_accuracy':.8630160692212608,'observed_gain':.02,'required_gain':.01,'dev_decisions':809},'temperature':{'temperature':1.3},'request':{'thinking_attempt':'abc123','native_checksums_sha256':hashlib.sha256((native/'checksums.json').read_bytes()).hexdigest(),'pool_checksums_sha256':hashlib.sha256((reverse/'checksums.json').read_bytes()).hexdigest(),'source_sha256':{},'input_sha256':{}}}
   if reject=='dev':fit['status']='dev_gate_failed'
   write(dev,{'result.json':fit})
   panel=[dict(id=str(i),suite='synthetic',group=str(i),state='public',question={'type':'noul','instructions':'True?'},label=True,source={}) for i in range(2111)]
   def mapped_path(value):
    if str(value).startswith('/results'):return results/Path(value).relative_to('/results')
    return Path(value)
   def scored(row,body):
    p=body['answers']['q']['noul'];return {'status':'ok','correct':p>.5,'probabilities':[p,1-p]}
   with patch.object(module,'Path',side_effect=mapped_path),patch('run_baselines.load_panel',return_value=panel) as load,patch('run_baselines.score',side_effect=scored),patch('run_baselines.record',side_effect=lambda row,value:dict(value,id=row['id'],suite=row['suite'])),patch('report_trained_joint.report',return_value={'macro_coverage_adjusted_accuracy':1.,'suites':{}}),patch('report_prediction_dominance.summarize',return_value={}):
    if reject:
     with self.assertRaises(ValueError):module.apply(dev,output)
     load.assert_not_called();self.assertFalse(output.exists());return
    value=module.apply(dev,output);load.assert_called_once();self.assertEqual(value['macro_coverage_adjusted_accuracy'],1.)
   rows=[json.loads(x) for x in (output/'calibrated-predictions.jsonl').read_text().splitlines()];self.assertEqual(len(rows),2111);self.assertTrue(all(r['correct'] for r in rows));self.assertTrue(all(abs(sum(r['probabilities'])-1)<1e-12 for r in rows));self.assertTrue((output/'checksums.json').exists());cfg=json.loads((output/'config.json').read_text());self.assertEqual(cfg['thinking_routed'],1);self.assertEqual(cfg['new_GPU_forward_passes'],0)
 def test_complete_cached_scoring_outputs_every_decision(self):self.exercise()
 def test_rejected_dev_never_reads_benchmark_panel(self):self.exercise('dev')
 def test_unverified_frozen_parent_never_reads_benchmark_panel(self):self.exercise('parameters')
if __name__=='__main__':unittest.main()
