"""Resume the exact official XR1 checkpoint through hf-mirror without proxies."""
from pathlib import Path
import hashlib,json,os,time,requests

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'models/official/model_states.pt'
SIZE=10226684862
SHA='94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7'
REV='ee21d524b5c52ac961d941e1bc7d6d92836c3d5e'
URL=f'https://hf-mirror.com/XiaomiRobotics/Xiaomi-Robotics-1-5B/resolve/{REV}/model_states.pt'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    for k in list(os.environ):
        if k.lower().endswith('_proxy'):os.environ.pop(k)
    os.environ.update(NO_PROXY='*',no_proxy='*')
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    state=TARGET.parent/'download.json';partial=TARGET.with_suffix('.partial');started=time.time()
    def update(stage,**changes):
        data=dict(stage=stage,pid=os.getpid(),file=str(TARGET),expected_sha256=SHA,revision=REV,
                  bytes=partial.stat().st_size if partial.exists() else TARGET.stat().st_size if TARGET.exists() else 0,
                  total=SIZE,elapsed_seconds=time.time()-started,proxies_used=False,**changes)
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(state)
    if TARGET.exists():
        assert TARGET.stat().st_size==SIZE and sha(TARGET)==SHA
        update('complete',verified=True);return
    session=requests.Session();session.trust_env=False
    for attempt in range(6):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset==SIZE:break
            update('downloading',attempt=attempt+1)
            with session.get(URL,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                if offset and response.status_code!=206:offset=0
                if offset:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                last=time.time()
                with partial.open('ab' if offset else 'wb') as out:
                    for chunk in response.iter_content(8<<20):
                        out.write(chunk)
                        if time.time()-last>5:update('downloading',attempt=attempt+1);last=time.time()
            break
        except requests.RequestException:
            if attempt==5:raise
            time.sleep(5)
    update('verifying')
    assert partial.stat().st_size==SIZE and sha(partial)==SHA
    partial.replace(TARGET);update('complete',verified=True)

if __name__=='__main__':main()
