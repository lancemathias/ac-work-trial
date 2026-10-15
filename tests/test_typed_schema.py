import copy,json,unittest
from pathlib import Path
from jsonschema import Draft202012Validator,ValidationError
ROOT=Path(__file__).resolve().parents[1]
class TrainingPointSchema(unittest.TestCase):
 def test_all_training_points_and_targets(self):
  outer=Draft202012Validator(json.loads((ROOT/'configs/training-point.schema.json').read_text()))
  for line in (ROOT/'data/training/typed-5000.jsonl').read_text().splitlines():
   row=json.loads(line);outer.validate(row);Draft202012Validator(row['model_payload']['schema']).validate(row['gold'])
 def test_unknown_confidence_field_and_invalid_target_rejected(self):
  row=json.loads((ROOT/'docs/training-point-example.json').read_text());outer=Draft202012Validator(json.loads((ROOT/'configs/training-point.schema.json').read_text()))
  mutated=copy.deepcopy(row);mutated['confidence']=.9
  with self.assertRaises(ValidationError):outer.validate(mutated)
  mutated=copy.deepcopy(row['gold']);mutated['decision']='nonexistent_option'
  with self.assertRaises(ValidationError):Draft202012Validator(row['model_payload']['schema']).validate(mutated)
if __name__=='__main__':unittest.main()
