#!/usr/bin/env python3
"""Wait for eight idle GPUs, validate full-size training, then run ten epochs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / '.queue/left_relative_10ep'
LAUNCHER = ROOT / 'xr1/scripts/train_left_8gpu.sh'
GLOBAL_BATCH_SIZE = int(os.environ.get('GLOBAL_BATCH_SIZE', '256'))
child = None


def gpu_status():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    return [dict(zip(('index','used_mib','util_percent'), map(int, line.split(','))))
            for line in result.stdout.strip().splitlines()]


def all_idle(gpus):
    return len(gpus)==8 and {g['index'] for g in gpus}==set(range(8)) and all(
        g['used_mib']<=1024 and g['util_percent']<=5 for g in gpus)


def write_status(state, **kwargs):
    data={'state':state,'pid':os.getpid(),'updated_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'hostname':socket.gethostname(),'machine':os.environ.get('XR1_MACHINE_LABEL'),
        'global_batch_size':GLOBAL_BATCH_SIZE,
        'policy':'8 GPUs <=1024 MiB used and <=5% utilization for 3 consecutive 60-second checks',**kwargs}
    temporary=QUEUE/'status.tmp'
    temporary.write_text(json.dumps(data,indent=2));temporary.replace(QUEUE/'status.json')
    print(json.dumps(data),flush=True)


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
    write_status('stopped',signal=signum)
    raise SystemExit(128+signum)


def execute(mode, micro):
    global child
    log_path=QUEUE/f'{mode}_mb{micro}.log'
    env=os.environ.copy();env['MICRO_BATCH']=str(micro)
    env['EXP_NAME']='left_relative_10ep_bs256_20261002'
    env['GLOBAL_BATCH_SIZE']=str(GLOBAL_BATCH_SIZE)
    write_status('smoke_running' if mode=='smoke' else 'training',micro_batch=micro,
        accumulation=GLOBAL_BATCH_SIZE//(8*micro),log=str(log_path))
    with log_path.open('a') as output:
        child=subprocess.Popen(['bash',str(LAUNCHER),mode],cwd=ROOT,env=env,
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        code=child.wait()
    child=None
    return code,log_path


def main(resume_smoke=False):
    QUEUE.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
        report=json.loads((ROOT/'xr1/data/ethernet_temporal_left_v1/prepared.json').read_text())
        assert report['active_action_dims']==list(range(7)) and report['quality_filtering'] is False
        assert (ROOT/'.setup/left_training_verified.json').exists(), 'Local checks have not passed'
        assert (ROOT/'.setup/swanlab_verified.json').exists(), 'SwanLab not verified'
        previous=QUEUE/'status.json'
        if previous.exists() and json.loads(previous.read_text()).get('state')=='completed':
            raise RuntimeError('This queue has already completed; choose a new run directory for another run')
        assert GLOBAL_BATCH_SIZE > 0 and GLOBAL_BATCH_SIZE % 8 == 0
        consecutive=0;micro=GLOBAL_BATCH_SIZE//8;smoke_passed=False
        if resume_smoke:
            import yaml
            smoke_dir=ROOT/f'xr1/outputs/project_xiaomi-robotics-1/left_relative_10ep_bs256_20261002_smoke_mb{micro}'
            completed=json.loads((smoke_dir/'completed.json').read_text())
            config=yaml.safe_load((smoke_dir/'config.yaml').read_text())
            assert completed['global_step']==1
            assert config['data']['params']['global_batch_size']==GLOBAL_BATCH_SIZE
            assert config['data']['params']['train_datasets']['batch_size']==micro
            assert config['trainer']['accumulate_grad_batches']==1
            smoke_passed=True
        while True:
            try:
                gpus=gpu_status()
            except Exception as exc:
                consecutive=0;write_status('waiting',gpu_query_error=str(exc));time.sleep(60);continue
            consecutive=consecutive+1 if all_idle(gpus) else 0
            if consecutive<3:
                write_status('waiting',gpus=gpus,idle_checks=consecutive,smoke_passed=smoke_passed,micro_batch=micro)
                time.sleep(60);continue
            mode='train' if smoke_passed else 'smoke'
            code,log=execute(mode,micro)
            if code==3: # Another job claimed a GPU between checking and launch.
                consecutive=0;continue
            if code:
                text=log.read_text(errors='replace')
                reason='CUDA OOM; global batch and no-accumulation settings preserved' if ('CUDA out of memory' in text or 'torch.OutOfMemoryError' in text) else 'Launcher exited with an error'
                write_status('failed',phase=mode,exit_code=code,reason=reason,micro_batch=micro,accumulation=1,log=str(log));return
            if mode=='smoke':
                smoke_passed=True;consecutive=0
                write_status('waiting',smoke_passed=True,micro_batch=micro,reason='full-size smoke passed; waiting for GPU cleanup')
                time.sleep(60);continue
            write_status('completed',micro_batch=micro,log=str(log));return


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-smoke',action='store_true',help='Reuse a completed smoke run with the same batch settings')
    main(parser.parse_args().resume_smoke)
