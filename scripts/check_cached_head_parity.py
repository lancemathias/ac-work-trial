"""GPU live/live and live/cached parity with the trained (nontrivial) E1 head."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from training_data import load_splits,digest
from train_joint import prepare,forward_frozen
from vendor.clef_joint_schema import JointSchemaHead,collate_records
from frozen_feature_cache import SparseOutputRows


def check(training,cache,manifest):
    training,cache=Path(training),Path(cache)
    cfg=json.loads((training/'config.json').read_text())['config']
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
    splits,_=load_splits(manifest);rows={r['id']:r for values in splits.values() for r in values}
    index=json.loads((cache/'index.json').read_text());selected=[]
    for kind in range(3):
        selected.append(next(r for r in index if any(q['question_type']==kind for q in r['questions'])))
    selected.append(max(index,key=lambda r:r['token_count']));selected=list({r['id']:r for r in selected}.values())
    m=json.loads((cache/'manifest.json').read_text());hashes={r['path']:r['sha256'] for r in m['files']}
    for name in set(r['shard'] for r in selected)|{'output-embedding-rows.safetensors'}:
        if digest(cache/name)!=hashes[name]:raise ValueError('Cache checksum changed')
    processor=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub')
    backbone=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir='/models/hub',dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'})
    backbone.requires_grad_(False).eval()
    head=JointSchemaHead(hidden_size=backbone.get_output_embeddings().weight.shape[1],**cfg['head']).float().cuda()
    head.load_state_dict(load_file(training/'best-head.safetensors',device='cuda'));head.train()
    lexical=load_file(cache/'output-embedding-rows.safetensors',device='cuda')
    lookup=SparseOutputRows(lexical['token_ids'],lexical['rows'])
    lexical_equal=torch.equal(backbone.get_output_embeddings().weight[lexical['token_ids']],lexical['rows'])
    results=[]
    for entry in selected:
        row,encoded=prepare([rows[entry['id']]],processor.tokenizer,cfg['max_input_tokens'])[0]
        batch=collate_records([encoded],processor.tokenizer.pad_token_id,torch.device('cuda'))
        with safe_open(cache/entry['shard'],framework='pt',device='cpu') as f:
            start=entry['token_offset'];end=start+entry['token_count']
            cached=f.get_slice('hidden_states')[start:end].cuda().unsqueeze(0)
        text=backbone.model.language_model if hasattr(backbone.model,'language_model') else backbone.model
        with torch.no_grad():live=text(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
        outputs=[];grads=None;differences=[];losses=[]
        for features,embedding in [(live,backbone.get_output_embeddings().weight),(live,backbone.get_output_embeddings().weight),(cached,lookup)]:
            head.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=head(features,batch['input_ids'],batch['attention_mask'],batch['records'],embedding)[0]
                loss=sum(torch.nn.functional.cross_entropy(z.float()[None],torch.tensor([q.option_ids.index(row['targets'][q.question_id])],device='cuda')) for q,z in zip(encoded.questions,logits,strict=True))
            outputs.append([z.detach().float().clone() for z in logits]);losses.append(float(loss.detach()))
            loss.backward()
            current={k:p.grad.detach().clone() for k,p in head.named_parameters() if p.grad is not None}
            if grads is None:grads=current
            else:differences.append(max(float((current[k]-g).abs().max()) for k,g in grads.items()))
        results.append(dict(id=row['id'],tokens=len(encoded.input_ids),features_equal=torch.equal(live,cached),feature_max_abs=float((live-cached).abs().max()),lexical_rows_equal=lexical_equal,
            live_repeat_logit_max_abs=max(float((a-b).abs().max()) for a,b in zip(outputs[0],outputs[1])),
            cached_logit_max_abs=max(float((a-b).abs().max()) for a,b in zip(outputs[0],outputs[2])),
            live_repeat_gradient_max_abs=differences[0],cached_gradient_max_abs=differences[1],losses=losses))
    return {'checkpoint_sha256':digest(training/'best-head.safetensors'),'probes':results,'interpretation':'Two live head passes measure backward nondeterminism; third uses stored BF16 features and sparse lexical rows. No optimizer updates.'}
