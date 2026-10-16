"""Matched live-versus-cached head training on CPU test-only fixtures."""
import copy,dataclasses,hashlib,json,sys,tempfile,unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_training import fixtures,CharacterTokenizer,TinyBackbone
from training_data import load_splits,canonical,digest
from train_joint import prepare,train_head
from frozen_feature_cache import FrozenFeatureBackbone
from vendor.clef_joint_schema import JointSchemaHead

class CachedTrainingTests(unittest.TestCase):
    def test_same_features_produce_identical_head_updates(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        cfg=dict(learning_rate=.01,weight_decay=0.,seed=7,epochs=2,warmup_fraction=.1,
            effective_batch_decisions=3,gradient_clip=1.,patience=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=fixtures(root);splits,_=load_splits(manifest)
            prepared={k:prepare(rows,CharacterTokenizer(),8192) for k,rows in splits.items()}
            live=TinyBackbone().to(torch.bfloat16).requires_grad_(False).eval()
            cache=root/'cache';cache.mkdir();index=[]
            for i,(row,encoded) in enumerate([x for rows in prepared.values() for x in rows]):
                ids=torch.tensor(encoded.input_ids)
                hidden=live.embedding(ids).detach().contiguous();name=f'shard-{i}.safetensors'
                save_file({'input_ids':ids,'hidden_states':hidden},str(cache/name))
                index.append(dict(id=row['id'],source=row['source'],group_id=row['group_id'],targets=row['targets'],
                    request_sha256=hashlib.sha256(canonical(row['request']).encode()).hexdigest(),
                    questions=[dataclasses.asdict(q) for q in encoded.questions],shard=name,token_offset=0,token_count=len(ids)))
            (cache/'index.json').write_text(json.dumps(index))
            save_file({'token_ids':torch.arange(129),'rows':live.embedding.weight.detach().contiguous()},str(cache/'output-embedding-rows.safetensors'))
            identity=dict(training_source_config=cfg,adapter_sha256='fixture',torch=str(torch.__version__),
                transformers=__import__('transformers').__version__,dtype='bfloat16',extraction_batch_size=1,no_truncation=True)
            m=dict(status='complete',cached_head_backward_verified=True,identity=identity,
                parity_probes=[dict(cached_vs_direct_head_max_abs_error=0)],
                files=[dict(path=p.name,sha256=digest(p)) for p in cache.iterdir()])
            (cache/'manifest.json').write_text(json.dumps(m));checksum=digest(cache/'manifest.json')
            cached=FrozenFeatureBackbone(cache,cfg,prepared,torch.device('cpu'),checksum,'fixture')
            h=JointSchemaHead(hidden_size=8,width=8,routing_layers=1,layers=1,heads=2,feedforward=16,dropout=0)
            heads=[copy.deepcopy(h),copy.deepcopy(h)];events=[[],[]]
            for i,backbone in enumerate([live,cached]):
                out=root/str(i);out.mkdir()
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    train_head(backbone,heads[i],prepared,0,torch.device('cpu'),cfg,out,events[i].append)
            for k,v in heads[0].state_dict().items():self.assertTrue(torch.equal(v,heads[1].state_dict()[k]),k)
            self.assertEqual([e['loss'] for e in events[0] if e['event']=='train_step'],[e['loss'] for e in events[1] if e['event']=='train_step'])
            cached.verify_unchanged()
            from vendor.clef_joint_schema import collate_records
            encodings=[encoded for values in prepared.values() for _,encoded in values]
            batch=collate_records(encodings[:4],0,torch.device('cpu'))
            features=cached.cached_features(batch)
            for i,encoded in enumerate(encodings[:4]):
                self.assertTrue(torch.equal(features[i,:len(encoded.input_ids)],live.embedding(torch.tensor(encoded.input_ids))))
            changed=copy.deepcopy(prepared);changed['train'][0][0]['targets']['yn']='false'
            with self.assertRaisesRegex(ValueError,'request or targets differ'):
                FrozenFeatureBackbone(cache,cfg,changed,torch.device('cpu'),checksum,'fixture')

    def test_four_record_batches_run_five_epochs_without_early_stopping(self):
        from unittest.mock import patch
        torch.set_num_threads(1);torch.manual_seed(7)
        cfg=dict(learning_rate=.01,weight_decay=0.,seed=7,epochs=5,warmup_fraction=.1,
            effective_batch_decisions=8,microbatch_records=4,early_stopping=False,gradient_clip=1.,patience=1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=fixtures(root);splits,_=load_splits(manifest)
            prepared={k:prepare(rows,CharacterTokenizer(),8192) for k,rows in splits.items()}
            head=JointSchemaHead(hidden_size=8,width=8,routing_layers=1,layers=1,heads=2,feedforward=16,dropout=0)
            events=[];out=root/'out';out.mkdir()
            cfg['selection_metric']='nll'
            with patch('train_joint.metrics',return_value={'nll':1.0}):
                train_head(TinyBackbone(),head,prepared,0,torch.device('cpu'),cfg,out,events.append)
            self.assertEqual([e['epoch'] for e in events if e['event']=='dev_epoch'],[1,2,3,4,5])
            self.assertEqual(json.loads((out/'best.json').read_text())['epoch'],1)


    def test_accuracy_selection_keeps_more_accurate_head_despite_worse_nll(self):
        from unittest.mock import patch
        torch.set_num_threads(1);torch.manual_seed(7)
        cfg=dict(learning_rate=.01,weight_decay=0.,seed=7,epochs=2,warmup_fraction=.1,
            effective_batch_decisions=3,early_stopping=False,gradient_clip=1.,patience=1,selection_metric='accuracy')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=fixtures(root);splits,_=load_splits(manifest)
            prepared={k:prepare(rows,CharacterTokenizer(),8192) for k,rows in splits.items()}
            head=JointSchemaHead(hidden_size=8,width=8,routing_layers=1,layers=1,heads=2,feedforward=16,dropout=0)
            events=[];out=root/'out';out.mkdir()
            values=[{'nll':1.,'accuracy':.3},{'nll':.4,'accuracy':.7},{'nll':.6,'accuracy':.8}]
            with patch('train_joint.metrics',side_effect=values):
                result=train_head(TinyBackbone(),head,prepared,0,torch.device('cpu'),cfg,out,events.append)
            best=json.loads((out/'best.json').read_text())
            self.assertEqual(best['epoch'],2);self.assertEqual(best['selection_metric'],'accuracy')
            self.assertEqual(result['best_dev_nll'],.6)
