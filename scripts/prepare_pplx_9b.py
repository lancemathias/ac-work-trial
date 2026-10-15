"""Sample exactly 10k published-corpus rows and build a real held-out Jev reference."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, '/project/scripts')
from run_baselines import score, load_panel
from autojev.evaluate import make_prediction, validate_coverage
from autojev.model import decision_messages
from transformers import AutoProcessor

SEED = 20260920
BASE = 'Qwen/Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    corpus = Path('/data/pplx-v1-public-reconstruction')
    out = Path('/data/pplx-v1-10k-seed20260920')
    out.mkdir(exist_ok=True)
    hashes = json.loads((corpus / 'checksums.json').read_text())
    for name in ['data/train.jsonl', 'data/dev.jsonl', 'data/temperature.jsonl']:
        if digest(corpus / name) != hashes[name]:
            raise ValueError('Reconstructed corpus bytes changed')
    read = lambda path: [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    all_train = read(corpus / 'data/train.jsonl')
    sample = random.Random(SEED).sample(all_train, 10000)
    folds = {'train': sample, 'development': read(corpus / 'data/dev.jsonl'),
             'temperature': read(corpus / 'data/temperature.jsonl')}
    panel = load_panel()
    normalized = lambda state: json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    test_states = {normalized(r['state']) for r in panel}
    test_ids = {r['id'] for r in panel}
    if any(r['id'] in test_ids or normalized(r['state']) in test_states for r in sample):
        raise ValueError('Sample overlaps the fixed benchmark panel')
    families = {}
    for role, rows in folds.items():
        families[role] = {(r['source']['dataset'], r['family']) for r in rows}
        for other in families:
            if other != role and families[role] & families[other]:
                raise ValueError('Family leakage across folds')
        path = out / (role + '.jsonl')
        payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        if path.exists() and path.read_text() != payload:
            raise ValueError('Existing sampled data differs')
        path.write_text(payload)
    # Retain a fixed, nonbenchmark probability diagnostic as in the released recipe.
    public = read(corpus / 'data/known.jsonl')[:10]
    (out / 'public.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in public))
    processor = AutoProcessor.from_pretrained(BASE, revision=REVISION, cache_dir='/models/hub', local_files_only=True)
    tokenizer = processor.tokenizer
    import itertools, string
    candidates = list(string.ascii_uppercase) + [''.join(x) for x in itertools.product(string.ascii_uppercase, repeat=2)]
    codes = [x for x in candidates if len(tokenizer.encode(x, add_special_tokens=False)) == 1][:255]
    test_lengths = []
    for row in panel:
        turns = decision_messages(row, codes)
        text = processor.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        test_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
    if max(test_lengths) > 8192:
        raise ValueError('PPLX prompt exceeds test input limit; no truncation')
    # Actual Jev predictions are required by upstream calibration eligibility gates.
    reference_path = out / 'jev-reference.jsonl'
    existing = {r['id']: r for r in read(reference_path)} if reference_path.exists() else {}
    import httpx
    def fetch(row):
        with httpx.Client(timeout=90.) as client:
            for attempt in range(5):
                response = client.post('https://api.typesafe.ai/v1/systemone',
                    headers={'Authorization': 'Bearer ' + os.environ['JEV_API_KEY']},
                    json={'model': 'jev-1.13.0', 'state': row['state'], 'questions': {'q': row['question']}})
                if response.status_code in [429, 500, 502, 503, 504] and attempt < 4:
                    time.sleep(min(10., 2 ** attempt))
                    continue
                response.raise_for_status()
                body = response.json()
                if body.get('model') != 'jev-1.13.0':
                    raise ValueError('Jev reference returned an unexpected model')
                values = score(row, body)
                prediction = make_prediction(row, values['probabilities'])
                prediction.update(model=body['model'], provider='typesafe.ai', request_sha256=hashlib.sha256(json.dumps({'state': row['state'], 'question': row['question']}, sort_keys=True).encode()).hexdigest())
                return prediction
    pending = [r for r in folds['development'] if r['id'] not in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, r) for r in pending]
        for future in concurrent.futures.as_completed(futures):
            value = future.result()
            with reference_path.open('a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            existing[value['id']] = value
            if len(existing) % 100 == 0:
                print(json.dumps({'event': 'jev_reference_progress', 'records': len(existing), 'total': len(folds['development'])}), flush=True)
    reference = [existing[r['id']] for r in folds['development']]
    validate_coverage(folds['development'], reference)
    reference_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reference))
    from collections import Counter
    manifest = {'version': 1, 'seed': SEED, 'rows': len(sample), 'corpus_rows': len(all_train),
        'sampling': 'Uniform sample without replacement from the full reconstructed train fold; held-out folds unchanged',
        'source_counts': dict(Counter(r['suite'] for r in sample)), 'image_rows': sum(bool(r.get('images')) for r in sample),
        'fold_counts': {k: len(v) for k, v in folds.items()}, 'train_ids': [r['id'] for r in sample],
        'source_train_sha256': hashes['data/train.jsonl'], 'backbone_id': BASE, 'backbone_revision': REVISION,
        'test_max_tokens': max(test_lengths), 'reference': 'Fresh Jev 1.13.0 via existing typesafe.ai account; original curated reference file is unpublished',
        'sha256': {p.name: digest(p) for p in out.glob('*.jsonl')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'manifest': manifest}), flush=True)


if __name__ == '__main__':
    main()
