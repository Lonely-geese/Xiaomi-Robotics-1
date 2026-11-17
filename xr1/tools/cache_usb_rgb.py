"""Build the pixel-exact USB uint8 NPY cache independently of Phoenix-MWA.

Input: source_inventory.json with parents containing uid, length and videos
(head, left wrist). Output: cache/source_inventory.json and three NPYs/episode.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('PYAV_LOGGING','off')
import av
import numpy as np

VIEWS=(('ego',None),('wrist_left','512:384:168:48'),
       ('wrist_right','256:256:296:112'))

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def cache_parent(job):
    parent,output,inventory_dir=job
    av.logging.restore_default_callback()
    cameras=[]
    for index,(name,crop) in enumerate(VIEWS):
        source=Path(parent['videos'][0 if index==0 else 1])
        if not source.is_absolute():source=Path(inventory_dir)/source
        source=source.resolve(strict=True)
        target=Path(output)/'rgb256'/parent['uid']/(name+'.npy')
        marker=target.with_suffix('.complete.json');length=int(parent['length'])
        identity=dict(source=str(source),source_sha256=sha256(source),frames=length,
                      crop=crop,shape=[length,256,256,3],dtype='uint8',
                      resize='ffmpeg_bicubic_256x256',crop_codec='libx264_qp0_yuv420p')
        if target.exists() or marker.exists():
            if not target.exists() or not marker.exists():
                raise RuntimeError(f'Incomplete existing cache: {target}; move it aside before rebuilding')
            saved=json.loads(marker.read_text())
            if any(saved.get(k)!=v for k,v in identity.items()) or sha256(target)!=saved.get('rgb_sha256'):
                raise RuntimeError(f'Existing cache identity differs: {target}; use a new output directory')
            data=np.load(target,mmap_mode='r',allow_pickle=False)
            assert list(data.shape)==identity['shape'] and data.dtype==np.uint8
            cameras.append(str(target));continue
        target.parent.mkdir(parents=True,exist_ok=True)
        video=source
        if crop:
            video=Path(output)/'cropped_videos'/parent['uid']/(name+'.mp4')
            video.parent.mkdir(parents=True,exist_ok=True)
            temporary=video.with_suffix('.partial.mp4')
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
                '-threads','1','-i',str(source),'-map','0:v:0','-an','-filter_threads','1',
                '-vf','crop='+crop,'-vsync','0','-c:v','libx264','-threads','1',
                '-preset','veryfast','-qp','0','-pix_fmt','yuv420p',str(temporary)],check=True)
            temporary.replace(video)
        temporary=target.with_suffix('.partial.npy')
        data=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,
                                      shape=(length,256,256,3))
        count=0
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            stream.thread_type='SLICE';stream.codec_context.thread_count=1
            for frame in container.decode(stream):
                if count>=length:raise ValueError(f'Extra video frames: {video}')
                data[count]=frame.reformat(width=256,height=256,format='rgb24',
                                            interpolation='BICUBIC').to_ndarray()
                count+=1
        if count!=length:raise ValueError(f'Video length mismatch: {video}: {count} != {length}')
        data.flush();del data
        temporary.replace(target)
        write_json(marker,identity|dict(rgb_sha256=sha256(target),complete=True))
        cameras.append(str(target))
    return dict(parent,cameras=cameras)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=32)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    inventory=args.inventory.resolve(strict=True);output=args.output.resolve()
    if output==inventory.parent:parser.error('Use a separate output directory')
    source=json.loads(inventory.read_text());parents=source['parents']
    if not parents:parser.error('Inventory has no parents')
    uids=[p['uid'] for p in parents]
    if len(set(uids))!=len(uids):parser.error('Episode uids must be unique')
    for parent in parents:
        uid=Path(parent['uid'])
        if uid.is_absolute() or '..' in uid.parts:parser.error('uid must stay inside output')
        if int(parent['length'])<=0 or len(parent['videos'])!=2:
            parser.error('Each parent requires positive length and two source videos')
    output.mkdir(parents=True,exist_ok=True)
    rows={};jobs=[(p,str(output),str(inventory.parent)) for p in parents]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(cache_parent,j) for j in jobs]
        for done,future in enumerate(as_completed(futures),1):
            row=future.result();rows[row['uid']]=row
            write_json(output/'progress.json',dict(completed=done,total=len(jobs)))
            print(f'RGB episodes {done}/{len(jobs)}',flush=True)
    result=dict(source,state='rgb_complete',source_inventory=str(inventory),
                parents=[rows[p['uid']] for p in parents])
    write_json(output/'source_inventory.json',result)

if __name__=='__main__':main()
