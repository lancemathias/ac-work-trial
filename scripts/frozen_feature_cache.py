"""Checked, GPU-resident frozen E1 features; no model-forward approximation."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from training_data import canonical,digest,mounted_path

class SparseOutputRows:
    def __init__(self,ids,rows):
        self.ids,self.rows=ids,rows
        self.shape=(int(ids.max())+1,rows.shape[1])
    def detach(self): return self
    def __getitem__(self,ids):
        positions=torch.searchsorted(self.ids,ids)
        if torch.any(positions>=len(self.ids)) or not torch.equal(self.ids[positions],ids):
            raise ValueError('Lexical token absent from cache')
        return self.rows[positions]

class FrozenFeatureBackbone(torch.nn.Module):
    def __init__(self,directory,cfg,prepared,device,expected_manifest_sha256,expected_adapter_sha256,expected_source_config=None,expected_extraction_batch_size=1):
        super().__init__()
        root=Path(directory)
        if digest(root/'manifest.json')!=expected_manifest_sha256:raise ValueError('Cache manifest changed')
        manifest=json.loads((root/'manifest.json').read_text());identity=manifest['identity']
        if manifest['status']!='complete' or not manifest.get('cached_head_backward_verified'):
            raise ValueError('Require complete backward-verified cache')
        if identity['training_source_config']!=(expected_source_config if expected_source_config is not None else cfg) or identity['adapter_sha256']!=expected_adapter_sha256:
            raise ValueError('Cache config/adapter differs from E1')
        if identity['torch']!=torch.__version__ or identity['transformers']!=__import__('transformers').__version__:
            raise ValueError('Runtime differs from E1 cache')
        if identity['dtype']!='bfloat16' or identity['extraction_batch_size']!=expected_extraction_batch_size or not identity['no_truncation']:
            raise ValueError('Cache extraction semantics differ')
        probes=manifest.get('parity_probes',[])
        if not probes or any(p['cached_vs_direct_head_max_abs_error']!=0 for p in probes):
            raise ValueError('Cache parity evidence missing or failed')
        files={entry['path']:entry for entry in manifest['files']}
        for name,entry in files.items():
            if digest(mounted_path(root,name))!=entry['sha256']:raise ValueError('Cache checksum mismatch: '+name)
        index=json.loads((root/'index.json').read_text());self.rows={r['id']:r for r in index}
        if len(self.rows)!=len(index):raise ValueError('Duplicate cache IDs')
        all_prepared=[item for rows in prepared.values() for item in rows]
        if {r['id'] for r,_ in all_prepared}!=set(self.rows):raise ValueError('Cache/split IDs differ')
        self.shards={}
        for name in sorted({r['shard'] for r in index}):
            if name not in files:raise ValueError('Unverified shard')
            loaded=load_file(str(root/name),device=str(device))
            for key,value in loaded.items():self.register_buffer(f'shard_{len(self._buffers)}',value)
            self.shards[name]=loaded
        if 'output-embedding-rows.safetensors' not in files:raise ValueError('Unverified lexical rows')
        lexical=load_file(str(root/'output-embedding-rows.safetensors'),device=str(device))
        self.register_buffer('lexical_ids',lexical['token_ids']);self.register_buffer('lexical_rows',lexical['rows'])
        self.output_rows=SparseOutputRows(self.lexical_ids,self.lexical_rows)
        for row,encoded in all_prepared:
            cached=self.rows[row['id']]
            if cached['targets']!=row['targets'] or cached['request_sha256']!=__import__('hashlib').sha256(canonical(row['request']).encode()).hexdigest():
                raise ValueError('Cache request or targets differ: '+row['id'])
            if cached['group_id']!=row['group_id'] or cached['source']!=row['source']:
                raise ValueError('Cache provenance differs')
            questions=[dict(question_id=q.question_id,question_type=q.question_type,question_span=list(q.question_span),option_spans=[list(x) for x in q.option_spans],option_ids=list(q.option_ids)) for q in encoded.questions]
            if questions!=cached['questions']:raise ValueError('Cache schema spans differ')
            shard=self.shards[cached['shard']];start=cached['token_offset'];end=start+cached['token_count']
            if tuple(shard['input_ids'][start:end].cpu().tolist())!=encoded.input_ids:raise ValueError('Cache token sequence differs')
            if not torch.isfinite(shard['hidden_states'][start:end]).all():raise ValueError('Nonfinite cached features')
        self.versions=[(value,value._version) for value in self.buffers()]
        self.provenance=dict(path=str(root),manifest_sha256=expected_manifest_sha256,identity=identity,
            parity_probes=probes,residency='GPU',cached_bytes=sum(t.numel()*t.element_size() for t in self.buffers()))
    def get_output_embeddings(self):return SimpleNamespace(weight=self.output_rows)
    def cached_features(self,batch):
        features=[]
        for record in batch['records']:
            r=self.rows[record.record_id];start=r['token_offset'];end=start+r['token_count']
            features.append(self.shards[r['shard']]['hidden_states'][start:end])
        padded=torch.nn.utils.rnn.pad_sequence(features,batch_first=True)
        if padded.shape[:2]!=batch['input_ids'].shape:raise ValueError('Cache batch shape mismatch')
        return padded
    def verify_unchanged(self):
        if any(t.requires_grad or t.grad is not None or t._version!=v for t,v in self.versions):
            raise RuntimeError('Cached frozen features changed')
