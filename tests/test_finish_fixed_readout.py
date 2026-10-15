import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
from finish_fixed_readout import best_epoch, verified_cache
from head_probe_io import seal, write_json


class FixedFinishTests(unittest.TestCase):
    def test_accuracy_selection_and_earliest_tie_ignores_better_nll(self):
        records = [{'epoch': 1, 'dev': {'accuracy': .8, 'nll': .2}},
                   {'epoch': 2, 'dev': {'accuracy': .9, 'nll': .5}},
                   {'epoch': 3, 'dev': {'accuracy': .9, 'nll': .1}}]
        self.assertEqual(best_epoch(records), 2)

    def test_parent_cache_is_verified_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'request.json', {'count': 1})
            write_json(root / 'result.json', {'status': 'complete'})
            torch.save({'features': torch.ones(1, 8), 'items': [{'id': 'a', 'option_ids': ['a', 'b']}]}, root / 'shard-000000.pt')
            seal(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            features, items = verified_cache(root)
            self.assertEqual(items[0]['id'], 'a')
            torch.testing.assert_close(features, torch.ones(1, 8))
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            with (root / 'shard-000000.pt').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaises(ValueError):
                verified_cache(root)


if __name__ == '__main__':
    unittest.main()
