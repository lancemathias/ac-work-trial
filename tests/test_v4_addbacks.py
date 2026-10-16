"""Check source label preservation and line-safe serialization."""
import json
import unittest
from prepare_v4_addbacks import normalize,canonical

class V4Conversion(unittest.TestCase):
    def row(self):
        return dict(split='train',variant='direct',source='wice',id='w1',group_id='g1',question_id='decision',license='cc-by-sa-4.0',license_use='commercial',state='text_A: evidence\ntext_B: claim',kind='choice',question='Assess support',options=['supported','not_supported','partially_supported'],target=[0.,0.,1.])
    def test_partial_support_stays_three_way(self):
        row=self.row();out,_=normalize(row,'x','hash',0)
        self.assertEqual(list(out['question']['criteria'].values()),row['options'])
        self.assertEqual(out['label'],'option_2')
        self.assertEqual(out['question']['type'],'choice')
    def test_shared_wice_evidence_groups_claims(self):
        a=self.row();b=self.row();b['state']='text_A: evidence\ntext_B: other claim'
        self.assertEqual(normalize(a,'x','h',0)[0]['group'],normalize(b,'x','h',1)[0]['group'])
    def test_soft_boolean_rejected(self):
        row=self.row();row.update(source='procedural-typed-decisions/evidence_sufficiency',kind='noul',question_id='has_conflict',options=[],target=[.25])
        with self.assertRaises(ValueError):normalize(row,'x','h',0)
    def test_unicode_preserved_without_breaking_jsonl(self):
        value={'state':'a\u2028b\u0085c\u2029d'}
        encoded=canonical(value)
        self.assertEqual(len(encoded.splitlines()),1)
        self.assertEqual(json.loads(encoded),value)
    def test_no_derived_variants(self):
        row=self.row();row['variant']='label_verification'
        self.assertIsNone(normalize(row,'x','h',0))

if __name__=='__main__':unittest.main()
