import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
from cached_v31 import combine


class V31Tests(unittest.TestCase):
    def fixture(self):
        items = [dict(id='choice', type='choice', option_ids=['a', 'b'], target_index=0),
                 dict(id='bool', type='noul', option_ids=['false', 'true'], target_index=1)]
        changed = [dict(id='choice', type='choice', option_ids=['b', 'a'], target_index=1),
                   dict(id='bool', type='noul', option_ids=['true', 'false'], target_index=0)]
        return (torch.tensor([[1., 2.], [3., 4.]]), items), (torch.tensor([[5., 6.], [7., 8.]]), changed)

    def test_only_one_choice_view_added_and_targets_groups_preserved(self):
        native, shuffled = self.fixture()
        merged, stats = combine(native, shuffled, {'choice': 'family1', 'bool': 'family2'})
        self.assertEqual(stats['training_views'], 3)
        self.assertEqual(stats['added_choice_views'], 1)
        torch.testing.assert_close(merged[0][-1], torch.tensor([5., 6.]))
        self.assertEqual(merged[1][-1]['group_id'], merged[1][0]['group_id'])
        self.assertEqual(merged[1][-1]['base_id'], 'choice')
        self.assertEqual(len({x['id'] for x in merged[1]}), 3)
        self.assertEqual(sum(x['type'] == 'noul' for x in merged[1]), 1)

    def test_wrong_semantic_target_rejected(self):
        native, shuffled = self.fixture()
        shuffled[1][0]['target_index'] = 0
        with self.assertRaises(ValueError):
            combine(native, shuffled, {'choice': 'family1', 'bool': 'family2'})

    def test_identity_permutation_retained_once_and_counted(self):
        native, _ = self.fixture()
        merged, stats = combine(native, copy.deepcopy(native), {'choice': 'family1', 'bool': 'family2'})
        self.assertEqual(stats['identity_permutations'], 1)
        self.assertEqual(len(merged[1]), 3)


if __name__ == '__main__':
    unittest.main()
