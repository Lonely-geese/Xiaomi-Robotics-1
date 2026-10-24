#!/usr/bin/env python3
"""Durable HG-DAgger launch: GPU smoke, then fresh full run with checkpoint recovery."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recap_recovery import latest_policy_checkpoint, recovery_adjustments


def main():
    repo = Path(__file__).resolve().parents[2]
    queue = repo/'.queue/hg_dagger_1563_20261021'
    queue.mkdir(parents=True, exist_ok=True)
    lock = (queue/'watchdog.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    base_name = 'hg_dagger_1563_10ep_bs256_20261021'
    attempts = 0
    error = ''
    adjusted = dict(workers=4, dit_checkpointing=False, disable_ib=False)
    while not (queue/'STOP').exists():
        stage = 'train' if (queue/'smoke_passed.json').exists() else 'smoke'
        name = base_name if stage == 'train' else base_name+'_smoke'
        output = repo/'xr1/outputs/project_xiaomi-robotics-1'/name
        if (output/'completed.json').exists():
            completed = json.loads((output/'completed.json').read_text())
            if stage == 'smoke':
                validation = json.loads((output/'validation_latest.json').read_text())
                if completed['global_step'] != 2 or validation['frames'] != dict(demo=7694, correction=673):
                    raise ValueError('Smoke did not complete two updates and full validation')
                (queue/'smoke_passed.json').write_text(json.dumps(dict(
                    output=str(output), validation=validation, completed=completed), indent=2))
                attempts, error = 0, ''
                continue
            (queue/'status.json').write_text(json.dumps(dict(stage='complete', state='completed',
                output=str(output), **completed), indent=2))
            return
        checkpoint = latest_policy_checkpoint(output) if stage == 'train' else None
        if checkpoint is None and (output/'run_record.json').exists():
            # No complete optimizer checkpoint: preserve the failed attempt and
            # start again from the requested initialization. Never overwrite it.
            output.rename(output.with_name(output.name+f'_incomplete_{time.time_ns()}'))
        updates = recovery_adjustments(error, attempts)
        adjusted['workers'] = min(adjusted['workers'], updates['workers'])
        adjusted['dit_checkpointing'] |= updates['dit_checkpointing']
        adjusted['disable_ib'] |= updates['disable_ib']
        env = dict(os.environ, EXP_NAME=name)
        if checkpoint:
            env['DAGGER_RESUME'] = str(checkpoint)
        else:
            env.pop('DAGGER_RESUME', None)
        if adjusted['disable_ib']:
            env['NCCL_IB_DISABLE'] = '1'
        command = ['bash', str(repo/'xr1/scripts/train_dagger_8gpu.sh'), 'train',
            f"data.params.train_datasets.num_workers={adjusted['workers']}",
            f"data.params.train_datasets.validation_workers={min(2, adjusted['workers'])}",
            f"+model.params.model.dit_gradient_checkpointing={str(adjusted['dit_checkpointing']).lower()}"]
        if stage == 'smoke':
            command += ['+smoke_test=true', '+trainer.enable_checkpointing=false']
        log_path = queue/f'{stage}.log'
        with log_path.open('a') as log:
            log.write('\n'+json.dumps(dict(attempt=attempts, command=command, resume=str(checkpoint)))+'\n')
            log.flush()
            process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status = dict(stage=stage, state='running', watchdog_pid=os.getpid(), child_pid=process.pid,
                started_at=time.time(), attempts=attempts, command=command,
                output=str(output), log=str(log_path), resumed_from=str(checkpoint) if checkpoint else None,
                adjustments=adjusted)
            (queue/'status.json').write_text(json.dumps(status, indent=2))
            code = process.wait()
        if code == 0:
            if not (output/'completed.json').exists():
                raise RuntimeError('Launcher exited without a completion record')
            continue
        with log_path.open('rb') as log:
            log.seek(max(0, log_path.stat().st_size - 40000))
            error = log.read().decode(errors='replace')
        attempts += 1
        status.update(state='retrying', exit_code=code, attempts=attempts, updated_at=time.time())
        (queue/'status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(dict(stage=stage, exit_code=code, retry=attempts)), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
