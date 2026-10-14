import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_training_splits import assign_roles
import torch
from train_joint import fit_temperature, forward_frozen, metrics, prepare, train_head, run
from training_data import canonical, digest, load_splits, target_option, validate_request, adapt_training_point
from vendor.clef_joint_schema import JointSchemaHead, collate_records
from jsonschema import Draft202012Validator, ValidationError

CONTRACT_PATH = Path(__file__).resolve().parents[1]/'configs/training-point.schema.json'
VALIDATOR = Draft202012Validator(json.loads(CONTRACT_PATH.read_text()))


def request():
    return dict(model='jev-latest', state={'text': 'a public labelled example'}, questions={
        'yn': dict(type='noul', instructions={'ask': 'Is the statement true?'}),
        'route': dict(type='choice', instructions='Which route?', criteria={'z': None, 'a': {'meaning': 'first'}}),
        'rating': dict(type='score', instructions='Rate severity', criteria=['low', 'medium', 'high'])})


def labels():
    return {'yn': dict(type='noul', noul=True), 'route': dict(type='choice', choice='z'),
            'rating': dict(type='score', score=1)}


class CharacterTokenizer:
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) % 127 + 1 for c in text])


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(129, 8)
    @property
    def model(self):
        return self
    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    def get_output_embeddings(self):
        return self.embedding


def point():
    return dict(id='ANLI:one', source='ANLI', family_id='ANLI:family',
        model_payload=dict(input=canonical({'state':{'text':'public labelled example'}, 'question':'Evaluate the state'}),
            schema={'type':'object', 'additionalProperties':False, 'required':['yn','route','rating'],
                'properties':{'yn':{'type':'boolean','description':'Is it true?'},
                    'route':{'type':'string','enum':['z','a'],'description':'z: payments; a: support'},
                    'rating':{'type':'string','enum':['0','1','2'],'description':'0 low, 1 medium, 2 high'}}}),
        gold={'yn':True, 'route':'z', 'rating':'1'}, field_types={'yn':'noul','route':'choice','rating':'score'},
        target_metadata={'annotation_fraction':.8}, transformation_trace=[{'operation':'native_recast','rationale':'AUDIT_ONLY'}],
        generator={'version':1,'seed':7,'seed_derivation':'test fixture'}, native_source={'split':'train'})


def fixtures(folder):
    manifest = dict(version=1, hygiene={'approved': True}, splits={}, training_point_schema_sha256=digest(CONTRACT_PATH))
    report = folder/'hygiene.json'
    report.write_text('{}')
    manifest['hygiene']['report'] = dict(path=report.name, sha256=digest(report))
    for role in ('train', 'dev', 'calibration'):
        row = point()
        row['id'], row['family_id'] = role, role
        row['model_payload']['input'] = canonical({'state':{'text':f'public labelled example {role}'},'question':'Evaluate the state'})
        input_file = folder/f'{role}.jsonl'
        input_file.write_text(canonical(row)+'\n')
        manifest['splits'][role] = dict(role=role, allowed_sources=['ANLI'], evaluation_only=False,
            points=dict(path=input_file.name, sha256=digest(input_file)))
    path = folder/'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


class DataTests(unittest.TestCase):
    def test_grouped_split_keeps_linked_families_together(self):
        points = []
        for index, family, state in [(0,'family-a','same'), (1,'family-b','same'), (2,'family-b','other')]:
            row = point()
            row['id'], row['family_id'] = str(index), family
            row['model_payload']['input'] = canonical(dict(state=state,question='question'))
            points.append(row)
        roles = assign_roles(points, 17)
        self.assertEqual(len(set(roles.values())), 1)
        self.assertEqual(roles, assign_roles(list(reversed(points)), 17))

    def test_jev_types_and_hard_outcomes(self):
        req = request()
        validate_request(req)
        self.assertEqual({k: target_option(q, labels()[k]) for k,q in req['questions'].items()},
                         {'yn':'true', 'route':'z', 'rating':'1'})
        with self.assertRaises(ValueError):
            target_option(req['questions']['rating'], dict(type='score', score=1.5))
        with self.assertRaises(ValueError):
            target_option(req['questions']['yn'], dict(type='noul', noul=0.8))
        with self.assertRaises(ValueError):
            validate_request(dict(req, answers=labels()))

    def test_manifest_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path, manifest = fixtures(folder)
            rows, _ = load_splits(path)
            self.assertNotIn('answers', rows['train'][0]['request'])
            train_file = folder/'train.jsonl'
            original = train_file.read_text()
            train_file.write_text(original+' ')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                load_splits(path)
            train_file.write_text(original)
            for mutation in ['group', 'state', 'evaluation_only', 'source', 'gold']:
                path, manifest = fixtures(folder)
                dev_file = folder/'dev.jsonl'
                dev = json.loads(dev_file.read_text())
                if mutation == 'group': dev['family_id'] = 'train'
                if mutation == 'state': dev['model_payload']['input'] = canonical({'state':{'text':'public labelled example train'},'question':'Evaluate the state'})
                if mutation == 'evaluation_only': manifest['splits']['dev']['evaluation_only'] = True
                if mutation == 'source': dev['source'] = 'ARC'
                if mutation == 'gold': dev['model_payload']['gold'] = dev['gold']
                dev_file.write_text(canonical(dev)+'\n')
                manifest['splits']['dev']['points']['sha256'] = digest(dev_file)
                path.write_text(json.dumps(manifest))
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, ValidationError)):
                    load_splits(path)

    def test_supplied_contract_adapter_excludes_all_audit_and_gold_fields(self):
        original = point()
        row = adapt_training_point(original, VALIDATOR)
        self.assertEqual(row['targets'], {'yn':'true', 'route':'z', 'rating':'1'})
        self.assertNotIn('AUDIT_ONLY', canonical(row['request']))
        self.assertNotIn('annotation_fraction', canonical(row['request']))
        self.assertNotIn('gold', canonical(row['request']))
        original['field_types']['yn'] = 'choice'
        with self.assertRaises(ValueError): adapt_training_point(original, VALIDATOR)

    def test_encoder_alignment_and_no_truncation(self):
        req = request()
        row = dict(id='one', request=req, targets={'yn':'true', 'route':'z', 'rating':'1'})
        _, encoded = prepare([row], CharacterTokenizer(), 8192)[0]
        self.assertEqual(encoded.questions[0].option_ids, ('true', 'false'))
        self.assertEqual(encoded.questions[1].option_ids, ('a', 'z'))
        self.assertEqual(encoded.questions[2].option_ids, ('0', '1', '2'))
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            prepare([row], CharacterTokenizer(), 10)


class LearningTests(unittest.TestCase):
    def test_entrypoint_writes_calibration_traces_and_verified_artifacts(self):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest, _ = fixtures(folder)
            cfg = json.loads((CONTRACT_PATH.parent/'e1-qwen35-9b.json').read_text())
            cfg.update(device='cpu', epochs=2, effective_batch_decisions=3, wandb_mode='disabled', cpu_threads=1)
            cfg.pop('dataset')  # Tiny fixture is not the production dataset.
            cfg['head'] = dict(width=8,routing_layers=1,layers=1,heads=2,feedforward=16,dropout=0.)
            config = folder/'config.json'
            config.write_text(json.dumps(cfg))
            with patch.object(AutoProcessor, 'from_pretrained', return_value=SimpleNamespace(tokenizer=CharacterTokenizer())), \
                 patch.object(Qwen3_5ForConditionalGeneration, 'from_pretrained', side_effect=lambda *a,**k:TinyBackbone()):
                result = run(config, manifest, folder/'run')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['temperature']['decisions'], 3)
            self.assertEqual(result['dev_raw']['accuracy'], result['dev_calibrated']['accuracy'])
            checksums = json.loads((folder/'run/checksums.json').read_text())
            self.assertTrue(all(digest(folder/'run'/name)==sha for name,sha in checksums.items()))
            traces = (folder/'run/dev-traces.jsonl').read_text().splitlines()
            self.assertEqual(len(traces),3)

    def test_temperature_improves_overconfidence_without_changing_argmax(self):
        items = [dict(logits=[5., 0.], target_index=int(i==0), type='choice') for i in range(10)]
        result = fit_temperature(items)
        self.assertGreater(result['temperature'], 1)
        self.assertLess(result['calibrated_nll'], result['raw_nll'])
        self.assertEqual(metrics(items)['accuracy'], metrics(items, result['temperature'])['accuracy'])

    def test_real_joint_head_learns_while_backbone_is_identical(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path, _ = fixtures(folder)
            splits, _ = load_splits(path)
            prepared = {k: prepare(v, CharacterTokenizer(), 8192) for k,v in splits.items()}
            backbone = TinyBackbone().requires_grad_(False)
            before = {k:v.clone() for k,v in backbone.state_dict().items()}
            head = JointSchemaHead(hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)
            cfg = dict(learning_rate=.01, weight_decay=0., seed=7, epochs=8, warmup_fraction=.1,
                       effective_batch_decisions=3, gradient_clip=1., patience=8)
            events = []
            result = train_head(backbone, head, prepared, 0, torch.device('cpu'), cfg, folder, events.append)
            self.assertTrue(result['head_changed'])
            self.assertTrue(all(torch.equal(v, before[k]) for k,v in backbone.state_dict().items()))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in backbone.parameters()))
            self.assertLess(result['best_dev_nll'], events[0]['nll'])
            self.assertTrue((folder/'best-head.safetensors').exists())
            # Batched mixed schemas have the same option dimensions and finite logits.
            batch = collate_records([prepared['train'][0][1], prepared['dev'][0][1]], 0, torch.device('cpu'))
            outputs = forward_frozen(backbone, head.eval(), batch)
            self.assertEqual([[len(z) for z in row] for row in outputs],
                             [[len(q.option_ids) for q in record.questions] for record in batch['records']])


if __name__ == '__main__':
    unittest.main()
