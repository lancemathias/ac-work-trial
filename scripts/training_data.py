"""Fail-closed E1 data loading. Gold answers never enter the model request."""
import hashlib
import json
import re
from pathlib import Path

ROLES = ('train', 'dev', 'calibration')


def validate_run_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid run ID')


def mounted_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path must be inside the mounted Volume')
    return path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True, allow_nan=False)


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def read_jsonl(path):
    return [parse_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def description(value, nullable=False):
    if not isinstance(value, (str, dict, list)) and not (nullable and value is None):
        raise ValueError('Description must be a string, object or array')
    canonical(value)  # Reject NaN/infinite/non-JSON content, including nested values.


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {'model', 'state', 'questions'}:
        raise ValueError('Request must contain only Jev model, state and questions fields')
    if not isinstance(request['model'], str) or not request['model']:
        raise ValueError('Missing API model name; actual backbone is selected by run config')
    description(request['state'])
    questions = request['questions']
    if not isinstance(questions, dict) or not questions:
        raise ValueError('At least one typed question is required')
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(question, dict):
            raise ValueError('Invalid question identifier/object')
        if not {'type', 'instructions'} <= set(question) or set(question) - {'type', 'instructions', 'criteria'}:
            raise ValueError('Invalid question fields')
        description(question['instructions'])
        kind, criteria = question['type'], question.get('criteria')
        if kind == 'noul':
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) - {'true', 'false'}:
                    raise ValueError('Noul criteria may describe true and false only')
                for value in criteria.values():
                    description(value)
        elif kind == 'choice':
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError('Choice needs 1..255 options')
            for key, value in criteria.items():
                if not isinstance(key, str) or not key:
                    raise ValueError('Choice option keys must be nonempty strings')
                description(value, nullable=True)
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError('Score needs 2..10 ordered levels')
            for value in criteria:
                description(value)
        else:
            raise ValueError(f'Unsupported question type: {kind}')


def target_option(question, answer):
    """Hard human labels only. A confidence or fractional score is not a target."""
    kind = question['type']
    field = {'noul': 'noul', 'choice': 'choice', 'score': 'score'}[kind]
    if not isinstance(answer, dict) or set(answer) != {'type', field} or answer['type'] != kind:
        raise ValueError('Gold answer must contain matching type and one hard outcome')
    value = answer[field]
    if kind == 'noul':
        if type(value) not in (bool, int, float) or value not in (0, 1):
            raise ValueError('Noul training target must be a boolean or 0/1 outcome')
        return 'true' if value else 'false'
    if kind == 'choice':
        if not isinstance(value, str) or value not in question['criteria']:
            raise ValueError('Gold choice is not an allowed option')
        return value
    if type(value) is not int or not 0 <= value < len(question['criteria']):
        raise ValueError('Gold score must be an observed integer level; no interpolation')
    return str(value)


def adapt_training_point(point, validator):
    """Use only model_payload for input; gold/audit metadata stay outside encoder."""
    validator.validate(point)
    from jsonschema import Draft202012Validator
    payload = point['model_payload']
    schema = payload['schema']
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(point['gold'])
    props = schema['properties']
    if set(props) != set(schema['required']) or set(props) != set(point['gold']) or set(props) != set(point['field_types']):
        raise ValueError('Schema, gold, required and field_types must have identical keys')
    content = parse_json(payload['input'])
    if not isinstance(content, dict) or set(content) != {'state', 'question'} or canonical(content) != payload['input']:
        raise ValueError('model_payload.input must be canonical JSON with state and question only')
    description(content['question'])
    questions, targets = {}, {}
    for qid, field in props.items():
        kind, gold = point['field_types'][qid], point['gold'][qid]
        instructions = dict(question=content['question'], field_description=field['description'])
        if kind == 'noul':
            if field['type'] != 'boolean' or type(gold) is not bool:
                raise ValueError('Noul must have boolean schema and hard boolean gold')
            questions[qid] = dict(type='noul', instructions=instructions)
            targets[qid] = 'true' if gold else 'false'
        elif kind == 'choice':
            if field['type'] != 'string':
                raise ValueError('Choice requires string enum schema')
            # Option descriptions are already part of the permitted field description.
            # Never recover them from audit-only transformation_trace.
            questions[qid] = dict(type='choice', instructions=instructions,
                                  criteria={option: None for option in field['enum']})
            targets[qid] = gold
        elif kind == 'score':
            if field['type'] != 'string' or not 2 <= len(field['enum']) <= 10:
                raise ValueError('Score requires 2..10 ordered string enum levels')
            questions[qid] = dict(type='score', instructions=instructions, criteria=field['enum'])
            targets[qid] = str(field['enum'].index(gold))
        else:
            raise ValueError('Unsupported field type')
    request = dict(model='jev-latest', state=content['state'], questions=questions)
    validate_request(request)
    return dict(id=point['id'], group_id=point['family_id'], source=point['source'],
                request=request, targets=targets)


def load_splits(manifest_path):
    from jsonschema import Draft202012Validator
    path = Path(manifest_path).resolve()
    manifest = parse_json(path.read_text())
    if manifest.get('version') != 1 or set(manifest.get('splits', {})) != set(ROLES):
        raise ValueError('Manifest must define exactly train, dev and calibration roles')
    hygiene = manifest.get('hygiene', {})
    if hygiene.get('approved') is not True or not isinstance(hygiene.get('report'), dict):
        raise ValueError('An approved, checksummed hygiene report is required')

    def verified(spec):
        file = (path.parent / spec['path']).resolve()
        if not file.is_relative_to(path.parent):
            raise ValueError('Data/report paths must stay inside the manifest directory')
        if digest(file) != spec['sha256']:
            raise ValueError(f'Hash mismatch: {file}')
        return file

    verified(hygiene['report'])
    contract_path = Path(__file__).resolve().parents[1]/'configs/training-point.schema.json'
    if manifest.get('training_point_schema_sha256') != digest(contract_path):
        raise ValueError('Manifest must pin the training-point schema hash')
    contract = parse_json(contract_path.read_text())
    Draft202012Validator.check_schema(contract)
    validator = Draft202012Validator(contract)
    seen_ids, seen_groups, seen_requests, seen_states = {}, {}, {}, {}
    splits = {}
    for role in ROLES:
        spec = manifest['splits'][role]
        if spec.get('role') != role or not spec.get('allowed_sources') or spec.get('evaluation_only') is not False:
            raise ValueError(f'Role/source authorization missing: {role}')
        inputs = read_jsonl(verified(spec['points']))
        if not inputs:
            raise ValueError(f'Empty split: {role}')
        rows = []
        for point in inputs:
            row = adapt_training_point(point, validator)
            if row['source'] not in spec['allowed_sources']:
                raise ValueError('Unauthorized source')
            for key in ('id', 'group_id', 'source'):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f'Missing provenance: {key}')
            rid = row['id']
            if rid in seen_ids:
                raise ValueError('Duplicate input ID')
            seen_ids[rid] = role
            for table, key in (
                (seen_groups, row['group_id']),
                (seen_requests, canonical({k: row['request'][k] for k in ('state', 'questions')})),
                (seen_states, canonical(row['request']['state'])),
            ):
                if key in table and table[key] != role:
                    raise ValueError('Group, exact state or request overlaps across data roles')
                table[key] = role
            rows.append(row)
        splits[role] = rows
    return splits, manifest
