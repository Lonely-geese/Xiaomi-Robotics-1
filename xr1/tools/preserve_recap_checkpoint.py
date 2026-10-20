#!/usr/bin/env python3
"""Preserve an exact-step model outside the rotating recovery checkpoint pool."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preserve(run, step):
    target = run/'milestones'/f'step{step}'
    if (target/'preserved.json').exists():
        saved = json.loads((target/'preserved.json').read_text())
        if int(saved['global_step']) != step:
            raise ValueError('Existing preserved checkpoint has the wrong step')
        return saved
    source = run/'recovery'/f'step{step}.ckpt'
    ready = source/'restart_ready.json'
    if not ready.exists():
        return None
    marker = json.loads(ready.read_text())
    if int(marker['global_step']) != step:
        raise ValueError('Checkpoint ready marker has the wrong training step')
    temporary = target.with_name(f'.step{step}.partial')
    if temporary.exists():
        shutil.rmtree(temporary)
    weights = temporary/'last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    weights.parent.mkdir(parents=True, exist_ok=True)
    original = source/'checkpoint/mp_rank_00_model_states.pt'
    digest = hashlib.sha256()
    with original.open('rb') as input_file, weights.open('wb') as output_file:
        for block in iter(lambda:input_file.read(8*1024*1024), b''):
            digest.update(block)
            output_file.write(block)
        output_file.flush()
        os.fsync(output_file.fileno())
    if checksum(weights) != digest.hexdigest():
        raise ValueError('Preserved model checksum does not match the source')
    import torch
    checkpoint = torch.load(weights, map_location='cpu', mmap=True, weights_only=False)
    saved_step = checkpoint.get('global_step', checkpoint.get('global_steps'))
    if int(saved_step) != step or 'model.recap_condition_embedding' not in checkpoint['module']:
        raise ValueError('Weights do not contain the requested RECAP policy step')
    del checkpoint
    for name in ('config.py','config.yaml','run_record.json'):
        if (run/name).exists():
            shutil.copyfile(run/name,temporary/name)
    if (run/'config.yaml').exists():
        import yaml
        config = yaml.safe_load((run/'config.yaml').read_text())
        normalization = config['data']['params']['train_datasets'].get('normalization')
        if normalization:
            shutil.copyfile(normalization,temporary/'normalize.json')
    saved = dict(global_step=step, epoch=marker['epoch'], world_size=marker['world_size'],
        source_checkpoint=str(source), model=str(target/'last.ckpt/checkpoint/mp_rank_00_model_states.pt'),
        sha256=digest.hexdigest(), size_bytes=weights.stat().st_size,
        preserved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='model weights, training configuration and normalization; optimizer shards are not copied',
        retention='independent copy outside the rotating recovery directory')
    (temporary/'preserved.json').write_text(json.dumps(saved,indent=2))
    temporary.rename(target)
    return saved


def main(args):
    args.state_dir.mkdir(parents=True,exist_ok=True)
    lock = (args.state_dir/f'step{args.step}.lock').open('w')
    try:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This checkpoint request already has a running watcher')
    request = args.state_dir/f'step{args.step}.json'
    state = dict(watcher_pid=os.getpid(),step=args.step,run_dir=str(args.run),state='waiting')
    while True:
        try:
            snapshot = preserve(args.run,args.step)
            state.update(state='saved' if snapshot else 'waiting',snapshot=snapshot)
            if snapshot:
                print(json.dumps(state),flush=True)
        except Exception as error:
            state.update(state='retrying',error=f'{type(error).__name__}: {error}')
            print(json.dumps(state),flush=True)
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = request.with_suffix('.partial.json')
        try:
            temporary.write_text(json.dumps(state,indent=2))
            temporary.replace(request)
        except OSError as error:
            state.update(state='retrying',error=f'{type(error).__name__}: {error}')
            print(json.dumps(state),flush=True)
        if state['state'] == 'saved':
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument('--run',type=Path,default=repo/'xr1/outputs/project_xiaomi-robotics-1/recap_10ep_bs256_20261020')
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--state-dir',type=Path,default=repo/'.queue/recap_20261020/checkpoint_requests')
    parser.add_argument('--poll-seconds',type=int,default=15)
    main(parser.parse_args())
