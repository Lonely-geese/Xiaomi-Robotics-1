#!/usr/bin/env python3
"""Download pinned RLinf backbone assets with resumable ranges and LFS SHA256 verification."""
import concurrent.futures as cf
import hashlib,json,os,time
from pathlib import Path
import requests
from huggingface_hub import get_token
import argparse
parser=argparse.ArgumentParser(description='Download pinned RLinf backbones through HF_ENDPOINT.')
parser.add_argument('--token-file',type=Path)
parser.add_argument('--use-proxy',action='store_true')
args=parser.parse_args()
repo_root=Path(__file__).resolve().parents[2]
os.chdir(repo_root)
if not args.use_proxy:
 for key in list(os.environ):
  if key.lower() in ('http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
secret=args.token_file.read_text().strip() if args.token_file and args.token_file.exists() else get_token()
if not secret:raise ValueError('A token authorized for google/gemma-3-270m is required')
endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
session=requests.Session();session.trust_env=args.use_proxy
print('endpoint',endpoint,'proxy_enabled',args.use_proxy,flush=True)
assets=[dict(repo='google/siglip2-so400m-patch14-224',revision='78e403963a4f6a3640d07803284752326fdf4edf',name='model.safetensors',size=4541963488,sha256='2041e5330b3db96299b26bc3a6506c7fa8d5e61d5c94f267dd8aa4cfb12c1050'),dict(repo='google/gemma-3-270m',revision='9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1',name='model.safetensors',size=536223056,sha256='abb58eb73aece5163624090d5383c59cbcdb9e9539ec84cb864f74386c4b21d5'),dict(repo='google/gemma-3-270m',revision='9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1',name='tokenizer.json',size=33384570,sha256='7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c')]
small_files={'google/siglip2-so400m-patch14-224':['config.json','preprocessor_config.json'],
             'google/gemma-3-270m':['config.json','tokenizer_config.json','special_tokens_map.json',
                                  'added_tokens.json','generation_config.json']}
for repo,names in small_files.items():
 revision=next(a['revision'] for a in assets if a['repo']==repo)
 folder=Path('models')/repo.split('/')[-1];folder.mkdir(parents=True,exist_ok=True)
 for name in names:
  for attempt in range(12):
   try:
    response=session.get(f'{endpoint}/{repo}/resolve/{revision}/{name}',
        headers={'Authorization':'Bearer '+secret},timeout=(10,30))
    response.raise_for_status();json.loads(response.content)
    temporary=folder/(name+'.partial');temporary.write_bytes(response.content);temporary.replace(folder/name)
    break
   except Exception:
    if attempt==11:raise
    time.sleep(2)
parts_root=Path('.setup/recap_20261020/model_download_parts');parts_root.mkdir(exist_ok=True)
chunk_size=4*1024*1024
jobs=[]
for asset in assets:
 asset['target']=str(Path('models')/asset['repo'].split('/')[-1]/asset['name'])
 asset['parts']=str(parts_root/(asset['repo'].split('/')[-1]+'_'+asset['name']))
 Path(asset['parts']).mkdir(exist_ok=True)
 # Resolve from the configured mirror, preserving already downloaded ranges.
 url=f"{endpoint}/{asset['repo']}/resolve/{asset['revision']}/{asset['name']}?download=true"
 for attempt in range(12):
  try:
   response=session.get(url,headers={'Authorization':'Bearer '+secret,'Range':'bytes=0-0'},stream=True,timeout=(10,20))
   response.raise_for_status();asset['url']=response.url;response.close()
   print('resolved_asset',asset['repo'],asset['name'],flush=True)
   break
  except Exception as e:
   print('resolve_retry',asset['repo'],type(e).__name__,flush=True)
   if attempt==11:raise
   time.sleep(2)
 for start in range(0,asset['size'],chunk_size):jobs.append((asset,start,min(start+chunk_size,asset['size'])))
print('pinned_download_assets', [{k:v for k,v in a.items() if k not in ('url','parts')} for a in assets],flush=True)
def download(job):
 asset,a,b=job;path=Path(asset['parts'])/str(a)
 if path.exists() and path.stat().st_size==b-a:return
 for attempt in range(12):
  try:
   with session.get(asset['url'],headers={'Range':f'bytes={a}-{b-1}'},stream=True,timeout=(15,45)) as r:
    r.raise_for_status()
    if r.status_code!=206 or r.headers.get('Content-Range')!=f"bytes {a}-{b-1}/{asset['size']}":
     # Resolve a fresh mirror URL when an intermediary cached another range.
     resolve_url=f"{endpoint}/{asset['repo']}/resolve/{asset['revision']}/{asset['name']}?download=true&range_start={a}"
     with session.get(resolve_url,headers={'Authorization':'Bearer '+secret,'Range':f'bytes={a}-{b-1}'},stream=True,timeout=(15,45)) as fresh:
      fresh.raise_for_status()
      if fresh.status_code!=206 or fresh.headers.get('Content-Range')!=f"bytes {a}-{b-1}/{asset['size']}":raise RuntimeError('Incorrect CDN byte range')
      with path.with_suffix('.partial').open('wb') as f:
       for data in fresh.iter_content(1024*1024):f.write(data)
    else:
     with path.with_suffix('.partial').open('wb') as f:
      for data in r.iter_content(1024*1024):f.write(data)
   if path.with_suffix('.partial').stat().st_size!=b-a:raise RuntimeError('Incomplete chunk')
   path.with_suffix('.partial').replace(path);return
  except Exception as e:
   if attempt==11:raise RuntimeError(f"Download failed: {asset['repo']} {asset['name']} offset {a}, {type(e).__name__}") from None
   time.sleep(min(10,attempt+1))
started=time.monotonic()
with cf.ThreadPoolExecutor(24) as pool:
 for count,_ in enumerate(cf.as_completed([pool.submit(download,j) for j in jobs]),1):
  _.result()
  if count%20==0:print(json.dumps(dict(chunks=count,total_chunks=len(jobs),seconds=round(time.monotonic()-started))),flush=True)
for asset in assets:
 target=Path(asset['target']);temporary=target.with_suffix(target.suffix+'.verified_partial');h=hashlib.sha256()
 with temporary.open('wb') as out:
  for start in range(0,asset['size'],chunk_size):
   data=(Path(asset['parts'])/str(start)).read_bytes();h.update(data);out.write(data)
 if h.hexdigest()!=asset['sha256']:
  parts=Path(asset['parts'])
  parts.rename(parts.with_name(parts.name+f'.bad_checksum_{time.time_ns()}'))
  raise RuntimeError('Official LFS SHA256 mismatch; affected range cache quarantined for retry')
 temporary.replace(target)
 print('verified_asset',asset['target'],h.hexdigest(),flush=True)
manifest=[{k:v for k,v in a.items() if k not in ('url','parts')} for a in assets]
Path('models/recap_rlinf_backbones.json').write_text(json.dumps(manifest,indent=2))
if args.token_file:args.token_file.unlink(missing_ok=True)
print('all_native_backbones_ready',flush=True)
