"""Read-only shared RAM cache. Missing entries fail rather than falling back to disk."""
import json,os,pickle
from functools import lru_cache
import numpy as np
from PIL import Image

@lru_cache(maxsize=1)
def manifest():
    with open(os.environ['XR1_RAM_CACHE']) as f:m=json.load(f)
    assert m['complete']
    return m

@lru_cache(maxsize=None)
def trajectory(path):
    with open(manifest()['trajectories'][str(path)],'rb') as f:return pickle.load(f)

@lru_cache(maxsize=None)
def sidecar(path):
    with open(manifest()['sidecars'][str(path)],'rb') as f:return pickle.load(f)

@lru_cache(maxsize=None)
def frames(path):
    r=manifest()['videos'][path]
    return np.memmap(r['ram_path'],mode='r',dtype=np.uint8,shape=tuple(r['shape']))

def images(traj,keys,frame):
    out=[]
    for key in keys:
        infos=traj
        for part in key.split('.'):infos=infos[part]
        for info in infos:
            assert info.get('crop_bbox') is None
            out.append(Image.fromarray(frames(info['path'])[max(frame,0)+int(info.get('start',0))]))
    return out
