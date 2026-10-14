import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('baselines', Path(__file__).parents[1] / 'scripts/run_baselines.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class BaselineTests(unittest.TestCase):
    def test_gold_never_sent(self):
        row = {'state': 'evidence', 'question': {'type': 'noul'}, 'label': True, 'source': {'gold': 'secret'}}
        self.assertEqual(set(b.payload(row, 'model')), {'model', 'state', 'questions'})

    def test_noul_confidence_and_metrics(self):
        r = b.score({'question': {'type': 'noul'}, 'label': False}, {'answers': {'q': {'type': 'noul', 'noul': .1}}})
        self.assertTrue(r['correct'])
        self.assertAlmostEqual(r['confidence'], .9)
        self.assertAlmostEqual(r['brier'], .02)

    def test_invalid_distribution_rejected(self):
        row = {'question': {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B'}}, 'label': 'a'}
        for probs in [{'a': .1, 'b': .1}, {'a': float('nan'), 'b': 1}, {'a': 1}]:
            with self.assertRaises(ValueError):
                b.score(row, {'answers': {'q': {'type': 'choice', 'probabilities': probs}}})

    def test_ordinal_argmax_is_not_rounded_expectation(self):
        r = b.score({'question': {'type': 'score', 'criteria': ['zero', 'one', 'two']}, 'label': 0},
                    {'answers': {'q': {'type': 'score', 'probabilities': {'0': .4, '1': .35, '2': .25}}}})
        self.assertEqual(r['prediction'], 0)
        self.assertAlmostEqual(r['score_mae'], .85)

    def test_frozen_panel(self):
        rows = b.load_panel()
        self.assertEqual(len(rows), 2111)
        self.assertTrue(all(r['evaluation_only'] for r in rows))
        self.assertEqual(len({r['id'] for r in rows}), len(rows))

if __name__ == '__main__': unittest.main()
