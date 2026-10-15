"""Cache assembly preserves live head updates and rejects changed replay data."""
import copy,dataclasses,hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from safetensors.torch import save_file
from compose_feature_cache import compose
from frozen_feature_cache import FrozenFeatureBackbone
from train_joint import prepare,train_head
from training_data import canonical,digest,load_splits
from vendor.clef_joint_schema import JointSchemaHead,collate_records
from test_training import fixtures,CharacterTokenizer,TinyBackbone

def make_cache(path,prepared,live,cfg):
 path.mkdir();index=[]
 for i,(row,encoded) in enumerate(prepared):
  ids=torch.tensor(encoded.input_ids);name=f'shard-{i}.safetensors'
  save_file({'input_ids':ids,'hidden_states':live.embedding(ids).detach().contiguous()},str(path/name))
  index.append({'id':row['id'],'source':row['source'],'group_id':row['group_id'],'targets':row['targets'],'request_sha256':hashlib.sha256(canonical(row['request']).encode()).hexdigest(),'questions':[dataclasses.asdict(q) for q in encoded.questions],'shard':name,'token_offset':0,'token_count':len(ids)})
 (path/'index.json').write_text(json.dumps(index));save_file({'token_ids':torch.arange(129),'rows':live.embedding.weight.detach().contiguous()},str(path/'output-embedding-rows.safetensors'))
 identity=dict(training_source_config=cfg,adapter_sha256='fixture',torch=str(torch.__version__),transformers=__import__('transformers').__version__,dtype='bfloat16',extraction_batch_size=1,no_truncation=True,backbone_id=cfg['backbone_id'],backbone_revision=cfg['backbone_revision'],schema_sha256='schema',encoder_sha256='encoder',hidden_layer='last',attention_implementation='sdpa',seed=7,hidden_size=8)
 m={'status':'complete','cached_head_backward_verified':True,'identity':identity,'parity_probes':[{'cached_vs_direct_head_max_abs_error':0}],'files':[{'path':p.name,'sha256':digest(p)} for p in path.iterdir()]}
 (path/'manifest.json').write_text(json.dumps(m));return {'directory':path,'manifest_sha256':digest(path/'manifest.json')}
class CompositionTests(unittest.TestCase):
 def fixture(self,root):
  torch.set_num_threads(1);torch.manual_seed(7)
  mp,_=fixtures(root);splits,_=load_splits(mp);prepared={k:prepare(v,CharacterTokenizer(),8192) for k,v in splits.items()}
  cfg=dict(backbone_id='fixture',backbone_revision='a'*40,dataset={'sha256':'b'*64},learning_rate=.01,weight_decay=0.,seed=7,epochs=2,warmup_fraction=.1,effective_batch_decisions=3,gradient_clip=1.,patience=2)
  live=TinyBackbone().to(torch.bfloat16).requires_grad_(False).eval();part=make_cache(root/'parent',[x for v in prepared.values() for x in v],live,cfg)
  return cfg,live,prepared,part
 def test_composed_and_aliased_features_preserve_updates(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cfg,live,prepared,part=self.fixture(root)
   row,encoding=prepared['train'][0];alias=copy.deepcopy(row);alias['id']='replay'
   prepared['train'].append((alias,dataclasses.replace(encoding,record_id='replay')))
   extra=copy.deepcopy(row);extra['id']='extra';extra['request']['questions']['yn']['instructions']='An alternate public question'
   encoded=prepare([extra],CharacterTokenizer(),8192);prepared['train'].extend(encoded)
   extra_part=make_cache(root/'extra',encoded,live,cfg);rows=[r for values in prepared.values() for r,_ in values]
   out=root/'composite';m=compose([part,extra_part],out,rows,cfg,{'replay':'train'})
   cached=FrozenFeatureBackbone(out,cfg,prepared,torch.device('cpu'),digest(out/'manifest.json'),'fixture')
   encodings=[e for values in prepared.values() for _,e in values];batch=collate_records(encodings,0,torch.device('cpu'));features=cached.cached_features(batch)
   for i,e in enumerate(encodings):self.assertTrue(torch.equal(features[i,:len(e.input_ids)],live.embedding(torch.tensor(e.input_ids))))
   h=JointSchemaHead(hidden_size=8,width=8,routing_layers=1,layers=1,heads=2,feedforward=16,dropout=0);heads=[copy.deepcopy(h),copy.deepcopy(h)];events=[[],[]]
   for i,backbone in enumerate([live,cached]):
    dest=root/str(i);dest.mkdir()
    with torch.autocast('cpu',dtype=torch.bfloat16):train_head(backbone,heads[i],prepared,0,torch.device('cpu'),cfg,dest,events[i].append)
   for k,value in heads[0].state_dict().items():self.assertTrue(torch.equal(value,heads[1].state_dict()[k]),k)
   self.assertEqual([e['loss'] for e in events[0] if e['event']=='train_step'],[e['loss'] for e in events[1] if e['event']=='train_step']);cached.verify_unchanged()
 def test_unselected_source_rows_are_not_exposed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cfg,live,prepared,part=self.fixture(root)
   extra=copy.deepcopy(prepared['train'][0][0]);extra['id']='unused-source';extra['request']['questions']['yn']['instructions']='Not part of this condition'
   extra_part=make_cache(root/'extra',prepare([extra],CharacterTokenizer(),8192),live,cfg)
   rows=[r for values in prepared.values() for r,_ in values];out=root/'filtered'
   compose([part,extra_part],out,rows,cfg,allow_source_subset=True)
   cached=FrozenFeatureBackbone(out,cfg,prepared,torch.device('cpu'),digest(out/'manifest.json'),'fixture')
   self.assertEqual(set(cached.rows),{r['id'] for r in rows});self.assertNotIn('unused-source',cached.rows)
   self.assertFalse(any(name.startswith('parts/1/') for name in cached.shards))
 def test_mixed_reference_and_optimized_kernels_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cfg,live,prepared,part=self.fixture(root)
   extra=copy.deepcopy(prepared['train'][0][0]);extra['id']='new-kernel-row'
   extra_part=make_cache(root/'extra',prepare([extra],CharacterTokenizer(),8192),live,cfg)
   path=root/'extra/manifest.json';manifest=json.loads(path.read_text())
   manifest['identity']['optimized_kernel_runtime']={'versions':{'fla-core':'0.5.2'}}
   path.write_text(json.dumps(manifest));extra_part['manifest_sha256']=digest(path)
   rows=[r for values in prepared.values() for r,_ in values]+[extra]
   with self.assertRaisesRegex(ValueError,'Kernel runtimes differ'):
    compose([part,extra_part],root/'mixed',rows,cfg)
 def test_tampering_and_changed_alias_targets_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cfg,live,prepared,part=self.fixture(root);rows=[r for values in prepared.values() for r,_ in values]
   alias=copy.deepcopy(rows[0]);alias['id']='replay';alias['targets']['yn']='false'
   with self.assertRaisesRegex(ValueError,'request or targets'):compose([part],root/'bad-alias',rows+[alias],cfg,{'replay':'train'})
   p=root/'parent/shard-0.safetensors';p.write_bytes(p.read_bytes()+b'changed')
   with self.assertRaisesRegex(ValueError,'checksum'):compose([part],root/'bad-bytes',rows,cfg)
if __name__=='__main__':unittest.main()
