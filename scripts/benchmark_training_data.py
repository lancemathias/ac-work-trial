"""Adapt normalized training records using the exact benchmark request wrapper."""
import hashlib,json

def digest(path):
    return hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest()

def adapt_training_point(point,validator):
    from training_data import validate_request,target_option
    validator.validate(point)
    request={'model':'jev-latest','state':point['state'],'questions':{'q':point['question']}}
    validate_request(request)
    kind=point['question']['type']
    target=target_option(point['question'],{'type':kind,kind:point['label']})
    return {'id':point['id'],'group_id':point['group'],'source':point['suite'],'request':request,'targets':{'q':target}}
