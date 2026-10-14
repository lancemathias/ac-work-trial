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

    def test_missing_trials_have_two_denominators(self):
        rows = [{'status': 'ok', 'correct': True, 'confidence': .8, 'nll': .2, 'brier': .1, 'latency_ms': 1},
                {'status': 'ok', 'correct': False, 'confidence': .8, 'nll': 1.6, 'brier': 1.2, 'latency_ms': 1},
                {'status': 'error'}]
        result = b.metrics(rows, expected=4)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['coverage_adjusted_accuracy'], .25)
        self.assertEqual(result['missed'], 2)
        self.assertEqual(result['disagreement_rate_valid'], .5)
        self.assertEqual(result['error_rate_missed_wrong'], .75)

    def test_table_only_duplicates_scores_with_misses(self):
        self.assertEqual(b.accuracy_cell(.5, .5, 4, 4), '50.00% (4/4)')
        self.assertEqual(b.accuracy_cell(.5, .25, 2, 4), '50.00% / 25.00% (2/4)')

if __name__ == '__main__': unittest.main()
