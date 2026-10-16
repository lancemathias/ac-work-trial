import argparse,json
from pathlib import Path
from . import FrozenDecider
p=argparse.ArgumentParser(description='Run the accepted frozen native/reverse decision recipe on a CUDA GPU')
p.add_argument('request',type=Path,help='JSON with state and question');p.add_argument('--cache-dir')
a=p.parse_args();request=json.loads(a.request.read_text());model=FrozenDecider.from_pretrained(a.cache_dir)
print(json.dumps(model.predict(request['state'],request['question']),indent=2))
