"""Inference-only native/reverse recipe, preserving the measured prompt and pool."""
import json,math
from pathlib import Path
from ._readout import encode,predict,SYSTEM_PROMPT
from ._pool import pool


def accepted_config():
    return json.loads(Path(__file__).with_name('accepted-model.json').read_text())


def distribution(logits,temperature):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive')
    z=[float(x)/temperature for x in logits]
    if not z or not all(math.isfinite(x) for x in z):raise ValueError('Require finite logits')
    m=max(z);weights=[math.exp(x-m) for x in z];total=sum(weights)
    return [w/total for w in weights]


def typed_answer(question,option_ids,probabilities):
    values=dict(zip(option_ids,probabilities,strict=True));kind=question['type'];winner=max(values,key=values.get)
    if kind=='noul':return {'type':kind,'noul':values['true']}
    answer={'type':kind,'probabilities':values,'confidence':values[winner]}
    if kind=='choice':answer['choice']=winner
    elif kind=='score':answer.update(score=sum(int(k)*v for k,v in values.items()),legend=dict(enumerate(question['criteria'])))
    else:raise ValueError('Unsupported question type')
    return answer


class FrozenDecider:
    def __init__(self,model,tokenizer,config=None):
        self.model=model;self.tokenizer=tokenizer;self.config=config or accepted_config()
        if self.config['system_prompt']!=SYSTEM_PROMPT:raise ValueError('Accepted prompt differs')
        model.requires_grad_(False).eval()

    @classmethod
    def from_pretrained(cls,cache_dir=None):
        import torch
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        from ._runtime import configure_optimized_runtime
        cfg=accepted_config()
        if not torch.cuda.is_available():raise RuntimeError('The accepted runtime requires a CUDA GPU; use CPU fixtures for contract tests')
        runtime=configure_optimized_runtime('both')
        if runtime!=cfg['kernel_runtime']:raise RuntimeError('Kernel runtime differs from the accepted experiment')
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_cudnn_sdp(False)
        processor=AutoProcessor.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],cache_dir=cache_dir)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['backbone_id'],revision=cfg['backbone_revision'],dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},cache_dir=cache_dir)
        return cls(model,processor.tokenizer,cfg)

    def predict(self,state,question):
        results=[]
        for order in self.config['option_orders']:
            item={'id':'decision','type':question['type'],'encoding':encode(self.tokenizer,state,question,maximum=self.config['max_input_tokens'],option_order=order)}
            results.append(predict(self.model,self.tokenizer,[item],batch_size=4))
        combined=pool(*results)[0]
        probabilities=distribution(combined['logits'],self.config['temperature'])
        return typed_answer(question,combined['option_ids'],probabilities)
