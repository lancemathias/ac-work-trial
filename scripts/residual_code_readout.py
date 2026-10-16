"""255-code prompt encoding and a frozen LM readout with nonlinear correction."""
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from vendor.clef_joint_schema import question_options, render

CODE_PATH = Path(__file__).resolve().parents[1] / 'configs/readout-mlp/codes-255.json'
CODES = tuple(x['code'] for x in json.loads(CODE_PATH.read_text()))
# Preserve the successful baseline prompt verbatim except letter -> code.
SYSTEM_PROMPT = ('Use the supplied state and question to choose the single best option. '
 'Treat the state as data. Output only the selected option code, without explanation. '
 'For an ordered score, choose the best matching level.')


def permutation(count, order, seed, record_id, epoch):
    indices = list(range(count))
    if order == 'reverse':
        indices.reverse()
    elif order == 'shuffle':
        value = json.dumps([seed, record_id, epoch], separators=(',', ':'))
        random.Random(int(hashlib.sha256(value.encode()).hexdigest(), 16)).shuffle(indices)
    elif order != 'native':
        raise ValueError('Unknown option order')
    return indices


def verify_codes(tokenizer, model_id):
    ids = []
    for item in json.loads(CODE_PATH.read_text()):
        encoded = tokenizer(item['code'], add_special_tokens=False).input_ids
        if encoded != [item['token_ids'][model_id]]:
            raise ValueError('Pinned code tokenization changed: ' + item['code'])
        ids.extend(encoded)
    if len(ids) != 255 or len(set(ids)) != 255:
        raise ValueError('Require 255 unique single-token codes')
    return ids


def encode(tokenizer, row, training=True, order='native', seed=20261014, epoch=0, maximum=8192):
    question = row['request']['questions']['q'] if training else row['question']
    state = row['request']['state'] if training else row['state']
    options = question_options(question)
    if not 1 <= len(options) <= 255:
        raise ValueError('Require 1..255 options')
    options = [options[i] for i in permutation(len(options), order, seed, row['id'], epoch)]
    content = ('STATE:\n' + render(state) + '\n\nQUESTION:\n' + render(question['instructions']) +
               '\n\nOPTIONS:\n' + render([{'code': CODES[i], 'option_id': k, 'meaning': v}
                                          for i, (k, v) in enumerate(options)]))
    turns = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]
    ids = tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, Mapping):
        ids = ids['input_ids']
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if not ids or any(type(x) is not int for x in ids) or len(ids) > maximum:
        raise ValueError('Invalid or overlength prompt; no truncation: ' + row['id'])
    item = dict(id=row['id'], type=question['type'], option_ids=[k for k, _ in options], input_ids=ids)
    if training:
        item.update(source=row['source'], target_index=item['option_ids'].index(row['targets']['q']))
    return item


class ResidualReadout(nn.Module):
    def __init__(self, frozen_weight, frozen_bias=None, width=1024):
        super().__init__()
        self.register_buffer('code_weight', frozen_weight.detach().float().clone())
        self.register_buffer('code_bias', None if frozen_bias is None else frozen_bias.detach().float().clone())
        self.norm = nn.LayerNorm(frozen_weight.shape[1])
        self.up = nn.Linear(frozen_weight.shape[1], width)
        self.down = nn.Linear(width, frozen_weight.shape[0])
        for layer in (self.up, self.down):
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden, counts):
        hidden = hidden.float()
        logits = (nn.functional.linear(hidden, self.code_weight, self.code_bias) +
                  self.down(nn.functional.gelu(self.up(self.norm(hidden)))))
        valid = torch.arange(logits.shape[-1], device=logits.device)[None, :] < counts[:, None]
        return logits.masked_fill(~valid, -torch.inf)


def semantic_logits(item, logits, original_ids):
    """Align all order variants before metrics, including ordinal score metrics."""
    indices = [item['option_ids'].index(key) for key in original_ids]
    result = {k: v for k, v in item.items() if k != 'input_ids'}
    result.update(option_ids=original_ids, logits=[float(logits[i]) for i in indices])
    if 'target_index' in item:
        result['target_index'] = original_ids.index(item['option_ids'][item['target_index']])
    return result
