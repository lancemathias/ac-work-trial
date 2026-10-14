import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from random_joint_baseline import RUN, SEEDS, merge_table, summarize
from run_baselines import ROOT, canonical, load_panel, record, score, sha


class RandomJointReportTests(unittest.TestCase):
    def test_three_seed_mean_and_table_are_not_ensemble(self):
        rows = load_panel()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / RUN
            out.mkdir()
            (out / 'config.json').write_text(json.dumps(dict(seeds=SEEDS,
                panel_sha256=sha(ROOT / 'data/manifests/baseline-panel.json'))))
            for index, seed in enumerate(SEEDS):
                seed_out = out / f'seed-{seed}'
                seed_out.mkdir()
                traces = []
                for row in rows:
                    q = row['question']
                    options = [False, True] if q['type']=='noul' else list(q['criteria']) if q['type']=='choice' else list(range(len(q['criteria'])))
                    # Two always-correct initializations and one always-wrong one.
                    target = row['label'] if index < 2 else next(x for x in options if x != row['label'])
                    ps = {str(x): float(x==target) for x in options}
                    answer = dict(type=q['type'])
                    if q['type']=='noul':
                        answer['noul'] = ps['True']
                    else:
                        answer['probabilities'] = ps
                    body = dict(answers={'q': answer})
                    traces.append(record(row, dict(score(row, body), response=body, seed=seed, latency_ms=1)))
                (seed_out/'predictions.jsonl').write_text(''.join(canonical(r)+'\n' for r in traces))
            result = summarize(base)
            self.assertAlmostEqual(result['overall']['mean'], 2/3)
            self.assertEqual(result['overall']['per_seed'], [1,1,0])
            (base/'results.md').write_text('| Benchmark | n | PPLX | Flash | Clef | Jev |\n|---|---:|---:|---:|---:|---:|\n| BBH | 200 | - | - | - | - |\n| **11-suite macro** | **2111** | - | - | - | - |\n\n')
            merge_table(base)
            once=(base/'results.md').read_text()
            self.assertIn('66.67%', once)
            merge_table(base)
            self.assertEqual(once,(base/'results.md').read_text())
            for line in once.splitlines():
                if line.startswith('|'):
                    self.assertEqual(len(line.split('|')),9)


if __name__=='__main__':
    unittest.main()
