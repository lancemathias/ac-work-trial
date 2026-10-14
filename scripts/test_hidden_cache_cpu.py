"""Small CPU fixtures for the cache reader; real GPU parity is checked in extraction."""
import json,pathlib,tempfile,torch
from safetensors.torch import save_file
from cache_hidden_runtime import HiddenCache,CachedEmbeddingLookup,digest,JointSchemaHead,EncodedRecord,EncodedQuestion

def run_tests():
 torch.manual_seed(20261014)
 with tempfile.TemporaryDirectory() as d:
  p=pathlib.Path(d);tokens=torch.arange(1,17,dtype=torch.long);hidden=torch.randn(16,8);embedding=torch.randn(17,8)
  save_file({'input_ids':tokens,'hidden_states':hidden},str(p/'s.safetensors'))
  save_file({'token_ids':tokens,'rows':embedding[1:]},str(p/'output-embedding-rows.safetensors'))
  q=EncodedQuestion('decision',1,(3,5),((5,7),(7,9)),('a','b'));r=EncodedRecord(tuple(tokens.tolist()),(q,),'fixture')
  index=[{'id':'fixture','source':'fixture','group_id':'family','targets':{'decision':'a'},'shard':'s.safetensors','token_offset':0,'token_count':16,'questions':[{'question_id':q.question_id,'question_type':q.question_type,'question_span':q.question_span,'option_spans':q.option_spans,'option_ids':q.option_ids}]}]
  (p/'index.json').write_text(json.dumps(index));manifest={'status':'complete','files':[{'path':n,'sha256':digest(p/n)} for n in ['s.safetensors','output-embedding-rows.safetensors','index.json']]};(p/'manifest.json').write_text(json.dumps(manifest))
  cache=HiddenCache(p,'cpu');batch=cache.get('fixture');assert torch.equal(batch['hidden_states'][0],hidden);assert torch.equal(batch['input_ids'][0],tokens);assert batch['records'][0]==r
  head=JointSchemaHead(hidden_size=8,width=8,routing_layers=1,layers=1,heads=2,feedforward=16,dropout=0).eval()
  direct=head(hidden.unsqueeze(0),tokens.unsqueeze(0),torch.ones_like(tokens).unsqueeze(0),[r],embedding)[0]
  cached=cache.forward(head,'fixture',False)[0];assert all(torch.equal(a,b) for a,b in zip(direct,cached))
  sum(v.square().mean() for v in cached).backward();assert any(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
  try:cache.output_embeddings[torch.tensor([100])];raise AssertionError('Unknown token accepted')
  except ValueError:pass
  (p/'s.safetensors').write_bytes(b'corrupt')
  try:HiddenCache(p,'cpu');raise AssertionError('Corrupt shard accepted')
  except ValueError:pass
  manifest['status']='partial';(p/'manifest.json').write_text(json.dumps(manifest))
  try:HiddenCache(p,'cpu');raise AssertionError('Partial cache accepted')
  except ValueError:pass
 return {'tensor_and_span_roundtrip':True,'cached_vs_direct_head_exact':True,'head_backward':True,'unknown_lexical_token_rejected':True,'corrupt_shard_rejected':True,'partial_cache_rejected':True}
