"""Acquire only the frozen candidate file allowlist; verify cached/downloaded bytes."""
import hashlib,json,pathlib,urllib.request,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--verify-only',action='store_true');args=a.parse_args()
 files=json.loads((ROOT/'configs/training-raw-files.json').read_text())
 for f in files:
  p=ROOT/f['path'];assert p.resolve().is_relative_to((ROOT/'data/training/raw').resolve())
  if not p.exists():
   if args.verify_only:raise FileNotFoundError(p)
   p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.download')
   with urllib.request.urlopen(f['url'],timeout=180) as r,temp.open('wb') as out:
    while b:=r.read(1048576):out.write(b)
   if digest(temp)!=f['sha256']:raise ValueError(f'Checksum mismatch: {temp}')
   temp.replace(p)
  assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256'],f"Pinned bytes mismatch: {p}"
 print(f'Verified {len(files)} pinned candidate files')
if __name__=='__main__':main()
