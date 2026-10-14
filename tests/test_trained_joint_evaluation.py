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
