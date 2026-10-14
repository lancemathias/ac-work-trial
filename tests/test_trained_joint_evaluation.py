import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_trained_joint import answer_body
from run_baselines import score

class EvaluationTests(unittest.TestCase):
    def test_boolean_and_choice_ordering(self):
        row={'label':True,'question':{'type':'noul'}}
        body=answer_body(row['question'],['true','false'],[.8,.2])
        self.assertTrue(score(row,body)['correct'])
        row={'label':'z','question':{'type':'choice','criteria':{'z':None,'a':None}}}
        body=answer_body(row['question'],['a','z'],[.1,.9])
        self.assertTrue(score(row,body)['correct'])
    def test_score_level_and_expected_value(self):
        row={'label':1,'question':{'type':'score','criteria':['low','medium','high']}}
        body=answer_body(row['question'],['0','1','2'],[.1,.7,.2])
        result=score(row,body)
        self.assertTrue(result['correct'])
        self.assertAlmostEqual(result['score_mae'],.1)

class ReportTests(unittest.TestCase):
    def test_calibrated_table_preserves_existing_columns_and_is_idempotent(self):
        import tempfile,json
        from run_baselines import ROOT,load_panel,sha,record,canonical
        from report_trained_joint import report
        rows=load_panel()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'config.json').write_text(json.dumps({'panel_sha256':sha(ROOT/'data/manifests/baseline-panel.json'),'temperature':.8,'checkpoint_sha256':'a'*64}))
            (out/'completion.json').write_text(json.dumps({'status':'complete','parameters_unchanged':True}))
            traces=[]
            for row in rows:
                q=row['question']
                options=['true','false'] if q['type']=='noul' else sorted(q['criteria']) if q['type']=='choice' else [str(i) for i in range(len(q['criteria']))]
                body=answer_body(q,options,[1/len(options)]*len(options))
                traces.append(record(row,dict(score(row,body),response=body,latency_ms=1)))
            (out/'calibrated-predictions.jsonl').write_text(''.join(canonical(r)+'\n' for r in traces))
            table=out/'comparison.md';original=(ROOT/'outputs/baselines-20261014/results.md').read_text();table.write_text(original)
            result=report(out,table);first=table.read_text();report(out,table)
            self.assertEqual(first,table.read_text())
            self.assertEqual(first.count('E1 Qwen3.5-9B calibrated'),1)
            self.assertEqual(result['temperature'],.8)
            for before,after in zip(original.splitlines(),first.splitlines()):
                if before.startswith('| BBH |'):
                    self.assertEqual([x.strip() for x in before.strip('|').split('|')],
                                     [x.strip() for x in after.strip('|').split('|')][:-1])
