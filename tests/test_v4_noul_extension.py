import unittest
from prepare_v4_noul_extension import condaqa,creak,multirc

class ExtensionConversion(unittest.TestCase):
    def conda(self,label):
        return dict(label=label,SampleID=1,PassageID=2,QuestionID='q',PassageEditID=0,sentence1='The door is not closed.',sentence2='Is the door closed?')
    def test_unknowns_filtered_before_selection(self):
        for label in ["DON'T KNOW",'UNKNOWN','free-form answer']:
            self.assertIsNone(condaqa(self.conda(label),{}))
    def test_no_and_yes_preserve_polarity(self):
        for label,target in [('YES',True),('NO',False)]:
            out=condaqa(self.conda(label),{})
            self.assertIs(out['label'],target)
            self.assertEqual(out['question'],{'type':'noul','instructions':'Is the door closed?'})
            self.assertEqual(out['state'],'The door is not closed.')
    def test_creak_explanation_excluded(self):
        r=dict(ex_id='train_0',sentence='A claim.',label='false',en_wiki_pageid=1,explanation='SECRET ANSWER RATIONALE')
        out=creak(r,{})
        self.assertFalse(out['label'])
        self.assertNotIn('SECRET',str(out))
        self.assertEqual(out['question']['type'],'noul')
    def test_multirc_same_parent_id_group_and_question(self):
        r=dict(idx={'paragraph':1,'question':2,'answer':3},label=1,paragraph='Passage',question='Question?',answer='Candidate')
        out=multirc(r,0,{})
        self.assertEqual(out['id'],'MultiRC:{"answer": 3, "paragraph": 1, "question": 2}')
        self.assertEqual(out['group'],'MultiRC:1')
        self.assertEqual(out['question']['instructions'],'Question? Is the candidate answer correct?')
        self.assertIs(out['label'],True)

if __name__=='__main__':unittest.main()
