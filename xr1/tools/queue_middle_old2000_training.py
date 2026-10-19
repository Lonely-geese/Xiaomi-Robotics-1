#!/usr/bin/env python3
"""Run the authorized mixed-data smoke check and five-epoch training durably."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / '.queue/middle_mixed_old2000_20261013'
LAUNCHER = ROOT / 'xr1/scripts/train_middle_mixed_8gpu.sh'
EXP_NAME = 'middle_mixed_old2000_5ep_lr2e6_20261013'
OUTPUT = ROOT / 'xr1/outputs/project_xiaomi-robotics-1' / EXP_NAME
child = None
status = {}


def record(state, **fields):
    global status
    status = {'state': state, 'controller_pid': os.getpid(),
              'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'experiment': EXP_NAME, 'output': str(OUTPUT), **fields}
    temporary = QUEUE / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(QUEUE / 'status.json')
    print(json.dumps(status), flush=True)


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    record('stopped', signal=signum)
    raise SystemExit(128 + signum)


def wait_idle():
    observations = 0
    while observations < 2:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True, timeout=20)
        gpus = [dict(zip(('index', 'used_mib', 'util_percent'),
                        (int(x.strip()) for x in line.split(','))))
                for line in result.stdout.strip().splitlines()]
        idle = (len(gpus) == 8 and {g['index'] for g in gpus} == set(range(8))
                and all(g['used_mib'] <= 1024 and g['util_percent'] <= 5 for g in gpus))
        observations = observations + 1 if idle else 0
        record('waiting_for_idle_gpus', idle_checks=observations, gpus=gpus)
        if observations < 2:
            time.sleep(15)


def execute(mode, micro):
    global child
    env = os.environ.copy()
    env.update(EXP_NAME=EXP_NAME, MICRO_BATCH=str(micro), GLOBAL_BATCH_SIZE='256',
               EPOCHS='5', MASTER_PORT='29715', LOG_BACKEND='swanlab', LOG_MODE='online',
               MIXED_DATA_DIR=str(ROOT/'xr1/data/ethernet_middle_mixed_old2000_20261013'))
    temporary = QUEUE / 'tmp'
    temporary.mkdir(exist_ok=True)
    env['TMPDIR'] = str(temporary)
    log = QUEUE / f'{mode}_mb{micro}.log'
    if log.exists():
        raise RuntimeError(f'Refusing to overwrite existing launch log: {log}')
    with log.open('x') as stream:
        child = subprocess.Popen(['bash', str(LAUNCHER), mode], cwd=ROOT / 'xr1',
                                 env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record('smoke_running' if mode == 'smoke' else 'training',
               child_pid=child.pid, micro_batch=micro,
               accumulation=256 // (8 * micro), log=str(log))
        while child.poll() is None:
            time.sleep(15)
        code = child.returncode
    child = None
    return code, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--micro-batch', type=int, default=32, choices=(32, 16, 8, 4))
    args = parser.parse_args()
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        if OUTPUT.exists():
            raise RuntimeError(f'Choose a new experiment name; output already exists: {OUTPUT}')
        prior = QUEUE / 'status.json'
        if prior.exists():
            raise RuntimeError('This queue has already been started; inspect its status before any retry')
        try:
            micro = args.micro_batch
            while True:
                wait_idle()
                code, log = execute('smoke', micro)
                if not code:
                    break
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 200000))
                    tail = stream.read().decode(errors='replace')
                oom = 'CUDA out of memory' in tail or 'torch.OutOfMemoryError' in tail
                if not oom or micro <= 4:
                    record('failed', phase='smoke', exit_code=code, log=str(log))
                    return code
                micro //= 2
                record('retrying_smoke_after_oom', micro_batch=micro,
                       global_batch_size=256, accumulation=256 // (8 * micro),
                       previous_log=str(log))
            record('smoke_passed', log=str(log))
            wait_idle()
            code, log = execute('train', micro)
            if code:
                record('failed', phase='train', exit_code=code, log=str(log))
                return code
            completion = OUTPUT / 'completed.json'
            if not completion.exists():
                raise RuntimeError('Training exited without its completion record')
            completed = json.loads(completion.read_text())
            if completed.get('epochs') != 5 or completed.get('test_completed'):
                raise RuntimeError(f'Unexpected completion record: {completed}')
            record('completed', log=str(log), completion=completed)
            return 0
        except Exception as exc:
            record('failed', reason=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
