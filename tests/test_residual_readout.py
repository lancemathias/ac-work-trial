import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
from residual_code_readout import CODES, ResidualReadout, encode, permutation, semantic_logits


class Tokenizer:
    def apply_chat_template(self, turns, **kwargs):
        self.turns = copy.deepcopy(turns)
        return list(turns[1]['content'].encode())


def example():
    return {'id': 'sample', 'source': 'fixture', 'request': {'state': 'Evidence', 'questions': {'q': {
        'type': 'choice', 'instructions': 'Which?', 'criteria': {'first': 'Alpha', 'second': 'Beta', 'third': 'Gamma'}}}},
        'targets': {'q': 'second'}}


class ResidualTests(unittest.TestCase):
    def test_random_init_adds_nonlinear_correction_and_masks_invalid_codes(self):
        torch.manual_seed(3)
        weight = torch.randn(255, 8)
        bias = torch.randn(255)
        head = ResidualReadout(weight, bias, width=16)
        hidden = torch.randn(3, 8)
        counts = torch.tensor([2, 26, 255])
        values = head(hidden, counts)
        self.assertGreater(head.down.weight.abs().sum(), 0)
        self.assertGreater(head.up.weight.abs().sum(), 0)
        self.assertTrue(torch.equal(head.up.bias, torch.zeros_like(head.up.bias)))
        self.assertTrue(torch.equal(head.down.bias, torch.zeros_like(head.down.bias)))
        for layer in [head.up, head.down]:
            bound = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
            self.assertLessEqual(float(layer.weight.detach().abs().max()), bound)
            self.assertLess(float(layer.weight.detach().min()), 0)
            self.assertGreater(float(layer.weight.detach().max()), 0)
        expected = torch.nn.functional.linear(hidden, weight, bias) + head.down(torch.nn.functional.gelu(head.up(head.norm(hidden))))
        for i, count in enumerate(counts):
            torch.testing.assert_close(values[i, :count], expected[i, :count], atol=0, rtol=0)
            self.assertTrue(torch.isneginf(values[i, count:]).all())
        self.assertEqual(dict(head.named_parameters()).keys(), {'norm.weight', 'norm.bias', 'up.weight', 'up.bias', 'down.weight', 'down.bias'})

    def test_training_learns_nonlinear_correction_without_changing_frozen_rows(self):
        torch.manual_seed(4)
        weight = torch.randn(255, 8)
        head = ResidualReadout(weight, width=16)
        before = head.code_weight.clone()
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
        hidden = torch.randn(6, 8)
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(hidden, torch.full((6,), 3)), torch.tensor([0, 1, 2, 0, 1, 2]))
            loss.backward()
            optimizer.step()
        self.assertGreater(head.up.weight.grad.abs().sum(), 0)
        torch.testing.assert_close(head.code_weight, before, atol=0, rtol=0)
        self.assertIsNone(head.code_weight.grad)

    def test_shuffle_is_reproducible_epoch_specific_and_target_stays_semantic(self):
        self.assertEqual(permutation(255, 'shuffle', 7, 'row', 1), permutation(255, 'shuffle', 7, 'row', 1))
        self.assertNotEqual(permutation(255, 'shuffle', 7, 'row', 1), permutation(255, 'shuffle', 7, 'row', 2))
        row = example()
        for epoch in range(20):
            item = encode(Tokenizer(), row, order='shuffle', epoch=epoch)
            self.assertEqual(item['option_ids'][item['target_index']], 'second')
            self.assertCountEqual(item['option_ids'], ['first', 'second', 'third'])

    def test_reversal_preserves_option_meanings_and_semantically_aligns_score(self):
        tokenizer = Tokenizer()
        item = encode(tokenizer, example(), order='reverse')
        self.assertEqual(item['option_ids'], ['third', 'second', 'first'])
        mapping = json.loads(tokenizer.turns[1]['content'].split('OPTIONS:\n')[1])
        self.assertEqual(mapping[0], {'code': 'A', 'option_id': 'third', 'meaning': 'Gamma'})
        result = semantic_logits(item, [3., 2., 1.], ['first', 'second', 'third'])
        self.assertEqual(result['logits'], [1., 2., 3.])
        self.assertEqual(result['target_index'], 1)

    def test_gold_never_enters_prompts_or_test_encodings(self):
        row = example()
        tokenizer = Tokenizer()
        encode(tokenizer, row)
        turns = tokenizer.turns
        row['targets']['q'] = 'first'
        encode(tokenizer, row)
        self.assertEqual(turns, tokenizer.turns)
        test = {'id': row['id'], 'state': row['request']['state'], 'question': row['request']['questions']['q'], 'label': 'SECRET'}
        result = encode(tokenizer, test, training=False, order='reverse')
        self.assertNotIn('target_index', result)
        self.assertNotIn('SECRET', str(tokenizer.turns))

    def test_all_255_slots_render_and_oversize_rejected(self):
        row = example()
        row['request']['questions']['q']['criteria'] = {f'key-{i}': f'meaning-{i}' for i in range(255)}
        row['targets']['q'] = 'key-254'
        tokenizer = Tokenizer()
        result = encode(tokenizer, row, maximum=100000)
        self.assertEqual(len(result['option_ids']), 255)
        self.assertEqual(result['option_ids'][result['target_index']], 'key-254')
        self.assertIn(CODES[254], tokenizer.turns[1]['content'])
        with self.assertRaises(ValueError):
            encode(tokenizer, row, maximum=10)

    def test_feature_extraction_selects_last_valid_position_and_resumes_verified_cache(self):
        from train_residual_readout import extract
        class Text(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 8).requires_grad_(False)
                self.calls = 0
            def forward(self, input_ids, **kwargs):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids).cumsum(1))
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Text()
        backbone = Backbone()
        prepared = [dict(id='a', input_ids=[1, 2, 3], option_ids=['x', 'y'], type='choice', target_index=0),
                    dict(id='b', input_ids=[4], option_ids=['y', 'x'], type='choice', target_index=1)]
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = SimpleNamespace(pad_token_id=0)
            features, items = extract(Path(directory)/'cache', prepared, backbone, tokenizer, lambda: None, float('inf'))
            self.assertEqual([x['id'] for x in items], ['a', 'b'])
            torch.testing.assert_close(features[0], backbone.model.embedding(torch.tensor([1, 2, 3])).sum(0))
            torch.testing.assert_close(features[1], backbone.model.embedding(torch.tensor(4)))
            calls = backbone.model.calls
            resumed, _ = extract(Path(directory)/'cache', prepared, backbone, tokenizer, lambda: None, float('inf'))
            self.assertEqual(backbone.model.calls, calls)
            torch.testing.assert_close(resumed, features)
            with (Path(directory)/'cache/shard-000000.pt').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError):
                extract(Path(directory)/'cache', prepared, backbone, tokenizer, lambda: None, float('inf'))

    def test_training_epoch_updates_head_and_order_metrics_use_semantic_alignment(self):
        from train_residual_readout import train_epoch, order_metrics
        torch.manual_seed(9)
        head = ResidualReadout(torch.randn(255, 8), width=16)
        items = [dict(id=str(i), option_ids=['x', 'y'], target_index=i % 2) for i in range(6)]
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
        cfg = dict(seed=7, effective_batch_decisions=3, warmup_fraction=0.1, learning_rate=0.01, gradient_clip=1.0)
        loss = train_epoch(head, optimizer, (torch.randn(6, 8), items), cfg, 0, 4)
        self.assertTrue(__import__('math').isfinite(loss))
        self.assertGreater(head.down.weight.abs().sum(), 0)
        native = [dict(id='a', option_ids=['x', 'y'], logits=[1., 2.])]
        self.assertEqual(order_metrics(native, native)['argmax_flip_rate'], 0)
        changed = [dict(id='a', option_ids=['x', 'y'], logits=[2., 1.])]
        self.assertEqual(order_metrics(native, changed)['argmax_flip_rate'], 1)


if __name__ == '__main__':
    unittest.main()
