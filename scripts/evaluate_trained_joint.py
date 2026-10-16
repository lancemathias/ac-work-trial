"""Evaluate a locked trained head on the existing frozen benchmark panel."""
import json
import math
import time
from pathlib import Path
import torch
from run_baselines import load_panel, payload, score, record, sha, canonical, write_json
from train_joint import forward_frozen
from vendor.clef_joint_schema import JointSchemaHead, encode_record, collate_records


def answer_body(question, option_ids, probabilities):
    distribution = dict(zip(option_ids, probabilities, strict=True))
    answer = {'type': question['type']}
    if question['type'] == 'noul':
        answer['noul'] = distribution['true']
    else:
        answer['probabilities'] = distribution
    return {'answers': {'q': answer}}


def evaluate(training, output, commit, checkpoint_callback=None):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from safetensors.torch import load_file
    training, output = Path(training), Path(output)
    receipt=json.loads((training/'summary.json').read_text())
    if receipt['status'] != 'complete' or not receipt['backbone_parameter_versions_unchanged']:
        raise ValueError('Require a completed frozen-backbone training run')
    hashes=json.loads((training/'checksums.json').read_text())
    for name in ['best-head.safetensors','temperature.json','config.json','best.json']:
        if sha(training/name)!=hashes[name]: raise ValueError('Checkpoint evidence changed: '+name)
    cfg=json.loads((training/'config.json').read_text())['config']
    temperature=json.loads((training/'temperature.json').read_text())['temperature']
    if not math.isfinite(temperature) or temperature<=0: raise ValueError('Invalid locked temperature')
    output.mkdir(parents=True,exist_ok=False)
    rows=load_panel()
    from run_baselines import ROOT
    write_json(output/'config.json',dict(training_run=training.name,checkpoint_sha256=hashes['best-head.safetensors'],
        best=json.loads((training/'best.json').read_text()),temperature=temperature,
        temperature_sha256=hashes['temperature.json'],panel_sha256=sha(ROOT/'data/manifests/baseline-panel.json'),
        backbone_id=cfg['backbone_id'],backbone_revision=cfg['backbone_revision'],
        head=cfg['head'],runner_sha256=sha(__file__),commit=commit,
        max_input_tokens=cfg['max_input_tokens'],expected=len(rows),
        protocol='Frozen existing 2111-example panel; no fitting/selection on benchmark labels; report raw and calibration-partition T'))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_cudnn_sdp(False)
    started=time.monotonic()
    try:
        from resident_backbone import get_processor,get_model
        processor=get_processor(cfg);backbone=get_model(cfg)
        backbone.requires_grad_(False).eval()
        from backbone_attention import install_attention_mode
        install_attention_mode(backbone,cfg.get('attention_mode','native_causal'))
        head=JointSchemaHead(hidden_size=backbone.get_output_embeddings().weight.shape[1],**cfg['head']).float().to('cuda')
        from head_runtime import configure_head
        configure_head(head,cfg)
        head.load_state_dict(load_file(training/'best-head.safetensors',device='cuda'))
        head.requires_grad_(False).eval()
        frozen=[(p,p._version) for p in list(backbone.parameters())+list(head.parameters())]
        for i,row in enumerate(rows,1):
            began=time.monotonic()
            try:
                encoded=encode_record(processor.tokenizer,payload(row,cfg['backbone_id']),max_length=10**9)
                if len(encoded.input_ids)>cfg['max_input_tokens']:
                    raise ValueError('Input exceeds common 8192-token limit; no truncation')
                batch=collate_records([encoded],processor.tokenizer.pad_token_id,torch.device('cuda'))
                with torch.no_grad():
                    logits=forward_frozen(backbone,head,batch,autocast=True)[0][0].float()
                    raw=logits.softmax(-1).cpu().tolist()
                    calibrated=(logits/temperature).softmax(-1).cpu().tolist()
                torch.cuda.synchronize()
                latency_ms=1000*(time.monotonic()-began)
                result={}
                for name,probs in [('raw',raw),('calibrated',calibrated)]:
                    body=answer_body(row['question'],encoded.questions[0].option_ids,probs)
                    result[name]=record(row,dict(score(row,body),response=body,latency_ms=latency_ms,input_tokens=len(encoded.input_ids)))
                del batch,logits
            except torch.cuda.OutOfMemoryError:
                raise  # Fail clearly rather than report an OOM-corrupted run as complete.
            except ValueError as exc:
                if 'no truncation' not in str(exc): raise
                result={name:record(row,dict(status='error',error=type(exc).__name__,detail=str(exc))) for name in ['raw','calibrated']}
            for name,value in result.items():
                with (output/f'{name}-predictions.jsonl').open('a') as stream:stream.write(canonical(value)+'\n')
            if i%50==0:
                print(f'Test evaluation: {i}/{len(rows)}; {time.monotonic()-started:.1f}s',flush=True)
                if checkpoint_callback:checkpoint_callback()
        unchanged=all(not p.requires_grad and p.grad is None and p._version==v for p,v in frozen)
        if not unchanged:raise RuntimeError('Evaluation modified model parameters')
        write_json(output/'completion.json',dict(status='complete',attempted=len(rows),parameters_unchanged=True,
            elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except Exception as exc:
        write_json(output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
    finally:
        write_json(output/'checksums.json',{str(p.relative_to(output)):sha(p) for p in output.iterdir() if p.is_file() and p.name!='checksums.json'})
        if checkpoint_callback:checkpoint_callback()
