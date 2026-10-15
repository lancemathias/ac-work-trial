"""Behavioral guards for training-only deterministic human-label views."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_binary_views import candidates,views
class BinaryViewTests(unittest.TestCase):
 def splits(self):
  train=[]
  for i,meaning in enumerate(['entailment','contradiction','neutral','entailment','contradiction','entailment']):
   train.append({'id':str(i),'group':'train-'+str(i),'suite':'ANLI','state':'train state '+str(i),'source':{'native_revision':'a'*40},'evaluation_only':False,'question':{'type':'choice','instructions':'Which relation?','criteria':{'x':meaning,'y':'other'}},'label':'x'})
  return {'train':train,'dev':[{'id':'dev','group':'dev','state':'held dev'}],'calibration':[{'id':'cal','group':'cal','state':'held calibration'}]}
 def test_balanced_human_labels_and_matched_replay(self):
  splits=self.splits();original=copy.deepcopy(splits);selected=candidates(splits,7,256)
  boolean=views(selected,'noul');control=views(selected,'choice_replay')
  self.assertEqual(len(boolean),4);self.assertEqual(sum(r['label'] for r in boolean),2)
  self.assertEqual([r['id'] for r in boolean],[r['id'] for r in control])
  for (parent,meaning),b,c in zip(selected,boolean,control):
   self.assertEqual(b['state'],parent['state']);self.assertEqual(b['group'],parent['group']);self.assertEqual(b['label'],meaning=='entailment')
   self.assertEqual(c['question'],parent['question']);self.assertEqual(c['label'],parent['label'])
  self.assertEqual(splits,original)
  splits['train'].reverse();self.assertEqual(selected,candidates(splits,7,256))
 def test_family_and_state_leakage_rejected(self):
  for field,value in [('group','dev'),('state','held calibration')]:
   splits=self.splits();splits['train'][0][field]=value
   with self.assertRaisesRegex(ValueError,'leakage'):candidates(splits,7,256)
if __name__=='__main__':unittest.main()
