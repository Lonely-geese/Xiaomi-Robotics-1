#!/usr/bin/env python3
"""Supervise the authorized return -> value -> advantage -> XR1 training run.

The process persists independently of the terminal. All stage transitions and
failures are recorded locally. Completed stages are immutable and reusable.
"""
import argparse
import csv
import datetime
import json
import math
import os
import resource
import traceback
from pathlib import Path
import subprocess
import tempfile
import time

from recap_recovery import latest_policy_checkpoint, process_alive, recovery_adjustments


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.repo = Path(__file__).resolve().parents[2]
        self.xr1 = self.repo/'xr1'
        self.native = getattr(args, 'profile', 'legacy') == 'rlinf_1563'
        run_name = 'recap_rlinf_1563_20261020' if self.native else 'recap_20261020'
        self.root = self.repo/'.queue'/run_name
        self.root.mkdir(parents=True, exist_ok=True)
        status_path = self.root/'status.json'
        self.previous_status = json.loads(status_path.read_text()) if status_path.exists() else None
        self.data = self.xr1/'data'/run_name
        self.value = self.xr1/'outputs'/('recap_rlinf_value_1563_20261020' if self.native else 'recap_value_20261020')
        exp_name = 'recap_rlinf_1563_10ep_bs256_20261020' if self.native else 'recap_10ep_bs256_20261020'
        self.policy = self.xr1/'outputs/project_xiaomi-robotics-1'/exp_name
        if self.native:
            os.environ.update(RECAP_DATA_DIR=str(self.data), EXP_NAME=exp_name, RECAP_CONFIG='recap_rlinf')
            os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
        self.status = dict(controller_pid=os.getpid(), started_at=timestamp(), stage='starting')
        self.settings_path = self.root/'recovery_settings.json'
        self.settings = json.loads(self.settings_path.read_text()) if self.settings_path.exists() else dict(
            workers=4, dit_checkpointing=False, disable_ib=False, attempts={})
        self.last_error_log = ''
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 65536), hard), hard))
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '1'
        os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '300'
        # Local IPC avoids NFS .nfs* cleanup races and AF_UNIX path limits.
        os.environ['TMPDIR'] = tempfile.mkdtemp(prefix='xr1-recap-', dir='/tmp')
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['NCCL_DEBUG'] = 'WARN'

    def record(self, **changes):
        self.status.update(changes, updated_at=timestamp())
        target = self.root/'status.json'
        temporary = target.with_suffix('.partial.json')
        temporary.write_text(json.dumps(self.status, indent=2))
        temporary.replace(target)
        print(json.dumps(self.status), flush=True)

    def wait_for_gpus(self):
        while True:
            result = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True, timeout=30)
            rows = [[int(v.strip()) for v in line.split(',')] for line in result.strip().splitlines()]
            if len(rows) == 8 and all(used <= 1024 and util <= 5 for used, util in rows):
                return
            self.record(state='waiting_for_idle_gpus', gpu_observation=rows)
            time.sleep(30)

    def stop_family(self, process):
        import psutil
        owned = []
        try:
            parent = psutil.Process(process.pid)
            owned = parent.children(recursive=True)+[parent]
        except psutil.NoSuchProcess:
            pass
        for identity in self.status.get('owned_processes', []):
            try:
                item = psutil.Process(identity['pid'])
                if abs(item.create_time()-identity['created']) < .01:
                    owned.append(item)
            except psutil.NoSuchProcess:
                pass
        for item in reversed(owned):
            try:
                item.terminate()
            except psutil.NoSuchProcess:
                pass
        def active(item):
            try:
                return item.is_running() and item.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False
        deadline = time.monotonic()+20
        remaining = [p for p in owned if active(p)]
        while remaining and time.monotonic()<deadline:
            time.sleep(.2)
            remaining = [p for p in remaining if active(p)]
        for item in remaining:
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass

    def command(self, stage, command):
        self.record(stage=stage, state='running', command=command, stage_started_at=timestamp())
        log = self.root/f'{stage}.log'
        self.last_error_log = str(log)
        environment = dict(os.environ)
        if self.settings['disable_ib']:
            environment['NCCL_IB_DISABLE'] = '1'
        if stage == 'policy_train':
            checkpoint = latest_policy_checkpoint(self.policy)
            if checkpoint:
                environment['RECAP_RESUME'] = str(checkpoint)
                self.record(resuming_from=str(checkpoint))
            elif self.policy.exists():
                self.archive_attempt(self.policy)
        if stage == 'policy_smoke':
            smoke = self.policy.with_name(self.policy.name+'_smoke')
            if smoke.exists():
                self.archive_attempt(smoke)
        if stage.startswith('policy_'):
            command = command+[f"data.params.train_datasets.num_workers={self.settings['workers']}"]
            if self.settings['dit_checkpointing']:
                command += ['+model.params.model.dit_gradient_checkpointing=true']
        with log.open('a') as output:
            output.write('\nSTAGE '+timestamp()+' '+json.dumps(command)+'\n')
            output.flush()
            process = subprocess.Popen(command, cwd=self.xr1, stdout=output,
                stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            self.record(child_pid=process.pid, log=str(log), owned_processes=[])
            if stage == 'prepare':
                (self.root/'prepare_process.json').write_text(json.dumps(dict(pid=process.pid, log=str(log))))
            last_change, last_size = time.monotonic(), -1
            while process.poll() is None:
                time.sleep(10)
                import psutil
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                    identities = [dict(pid=p.pid,created=p.create_time()) for p in children]
                    if identities != self.status.get('owned_processes'):
                        self.record(owned_processes=identities)
                except psutil.NoSuchProcess:
                    pass
                size = log.stat().st_size
                if size != last_size:
                    last_change, last_size = time.monotonic(), size
                if time.monotonic()-last_change > self.args.stall_seconds:
                    self.stop_family(process)
                    raise RuntimeError(f'{stage}: no output for {self.args.stall_seconds}s; retrying from checkpoint')
            returncode = process.returncode
        if returncode:
            self.stop_family(process)
            raise RuntimeError(f'{stage} exited {returncode}; see {log}')
        self.record(last_completed_stage=stage, stage_completed_at=timestamp())

    def archive_attempt(self, root):
        target = root.with_name(root.name+f'.attempt_{time.time_ns()}')
        root.rename(target)
        self.record(archived_attempt=str(target))

    def run_forever(self):
        while True:
            try:
                self.run()
                return
            except Exception as error:
                stage = self.status.get('stage', 'starting')
                attempts = self.settings['attempts']
                attempts[stage] = attempts.get(stage, 0)+1
                detail = traceback.format_exc()
                if self.last_error_log and Path(self.last_error_log).exists():
                    with Path(self.last_error_log).open('rb') as handle:
                        handle.seek(max(0, Path(self.last_error_log).stat().st_size-32768))
                        detail += '\n'+handle.read().decode(errors='replace')
                fixes = recovery_adjustments(detail, attempts[stage])
                if stage == 'policy_train' and any(word in detail.lower() for word in (
                        'pytorchstreamreader', 'unexpected eof', 'failed finding central directory')):
                    checkpoint = latest_policy_checkpoint(self.policy)
                    if checkpoint:
                        marker = checkpoint/'restart_ready.json'
                        marker.rename(marker.with_name(f'restart_corrupt_{time.time_ns()}.json'))
                self.settings['workers'] = min(self.settings['workers'], fixes['workers'])
                self.settings['dit_checkpointing'] |= fixes['dit_checkpointing']
                self.settings['disable_ib'] |= fixes['disable_ib']
                temporary = self.settings_path.with_suffix('.partial.json')
                temporary.write_text(json.dumps(self.settings, indent=2))
                temporary.replace(self.settings_path)
                diagnostic = self.root/f"error_{stage}_{attempts[stage]:03d}.txt"
                diagnostic.write_text(detail)
                delay = min(600, 15*2**min(attempts[stage]-1, 6))
                self.record(state='recovering', error=f'{type(error).__name__}: {error}',
                    error_diagnostic=str(diagnostic), retry_in_seconds=delay,
                    recovery_settings=self.settings)
                time.sleep(delay)

    def run(self):
        self.adopt_orphan_stage()
        prepare_script = 'tools/prepare_recap_rlinf.py' if self.native else 'tools/prepare_recap_dataset.py'
        # Attach to the already launched preparation, without a competing writer.
        if not (self.data/'prepared.json').exists():
            process_file = self.root/'prepare_process.json'
            if not process_file.exists():
                self.command('prepare', ['python','-u',prepare_script])
            else:
                pid = json.loads(process_file.read_text())['pid']
                if not process_alive(pid):
                    self.command('prepare', ['python','-u',prepare_script])
                    pid = json.loads(process_file.read_text())['pid']
                self.record(stage='prepare', state='running', child_pid=pid,
                    log=str(self.root/'prepare.log'))
                self.last_error_log = str(self.root/'prepare.log')
                last_progress, last_changed = None, time.monotonic()
                while not (self.data/'prepared.json').exists():
                    try:
                        state = Path(f'/proc/{pid}/stat').read_text().split()[2]
                    except FileNotFoundError:
                        state = 'exited'
                    if state in ('Z','exited'):
                        raise RuntimeError('Dataset preparation stopped before publishing prepared.json')
                    progress = self.preparation_progress()
                    if progress != last_progress:
                        last_progress, last_changed = progress, time.monotonic()
                    if time.monotonic()-last_changed > self.args.stall_seconds:
                        import psutil
                        parent = psutil.Process(pid)
                        self.stop_family(parent)
                        raise RuntimeError('Preparation stalled; retrying cached episodes')
                    self.record(preparation_progress=progress)
                    time.sleep(30)
        if self.native and not (self.repo/'models/recap_rlinf_backbones.json').exists():
            self.command('backbones', ['python','-u','tools/download_recap_backbones.py',
                '--token-file','/tmp/xr1-recap-hf-token','--use-proxy'])
        if self.native:
            value_smoke = self.value.with_name(self.value.name+'_smoke')
            if not (value_smoke/'completed.json').exists():
                self.wait_for_gpus()
                self.command('value_smoke', ['torchrun','--standalone','--nproc_per_node=8',
                    'tools/train_recap_rlinf_value.py','--steps','2','--output',str(value_smoke),
                    '--workers',str(self.settings['workers']),'--resume'])
        if not (self.value/'completed.json').exists():
            self.wait_for_gpus()
            command = ['tools/train_recap_rlinf_value.py','--steps','8000'] if self.native else [
                'tools/train_recap_value.py','--epochs',str(self.args.value_epochs)]
            self.command('value_train', ['torchrun','--standalone','--nproc_per_node=8']+command+[
                '--resume','--workers',str(self.settings['workers'])])
        if not (self.data/'advantages_ready.json').exists():
            self.wait_for_gpus()
            self.command('advantages', ['torchrun','--standalone','--nproc_per_node=8',
                'tools/compute_recap_advantages.py','--data',str(self.data),
                '--checkpoint',str(self.value/'best.pt'),'--batch-size','32' if self.native else '256'])
        smoke = self.policy.with_name(self.policy.name+'_smoke')
        if not (smoke/'completed.json').exists():
            self.wait_for_gpus()
            self.command('policy_smoke', ['bash','scripts/train_recap_8gpu.sh','smoke'])
        self.verify_smoke(smoke)
        if not (self.policy/'completed.json').exists():
            self.wait_for_gpus()
            self.command('policy_train', ['bash','scripts/train_recap_8gpu.sh','train'])
        completion = json.loads((self.policy/'completed.json').read_text())
        manifest = json.loads((self.data/'prepared.json').read_text())
        expected = max(manifest['sft_frames'],manifest['rollout_frames'])//128*10
        if completion['epochs'] != 10 or completion['global_step'] != expected:
            raise RuntimeError(f'Unexpected policy completion: {completion}, expected {expected} updates')
        if not (self.policy/'last.ckpt/checkpoint/mp_rank_00_model_states.pt').exists():
            raise RuntimeError('Final model checkpoint missing')
        self.record(stage='complete', state='completed', completion=completion,
            final_checkpoint=str(self.policy/'last.ckpt/checkpoint/mp_rank_00_model_states.pt'))

    def adopt_orphan_stage(self):
        previous, self.previous_status = self.previous_status, None
        if not previous or previous.get('stage') not in ('backbones','value_smoke','value_train','advantages','policy_smoke','policy_train'):
            return
        pid = previous.get('child_pid')
        if not pid:
            return
        if not process_alive(pid):
            from types import SimpleNamespace
            self.status['owned_processes'] = previous.get('owned_processes', [])
            self.stop_family(SimpleNamespace(pid=pid))
            return
        import psutil
        parent = psutil.Process(pid)
        expected_cwd = self.repo if previous.get('stage') == 'backbones' else self.xr1
        if str(expected_cwd) != parent.cwd():
            return
        stage, log = previous['stage'], Path(previous['log'])
        self.last_error_log = str(log)
        self.record(stage=stage, state='reattached', child_pid=pid, log=str(log))
        last_size, last_change = log.stat().st_size, time.monotonic()
        while process_alive(pid):
            time.sleep(10)
            size = log.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.monotonic()
            if time.monotonic()-last_change > self.args.stall_seconds:
                self.stop_family(parent)
                raise RuntimeError(f'Reattached {stage} stalled; recovering from checkpoint')

    def preparation_progress(self):
        path = self.root/'prepare.log'
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size-8192))
            lines = handle.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            if line.startswith('{"processed"'):
                return json.loads(line)
        return {'state':'indexing SFT segments / preparing first batch'}

    def verify_smoke(self, smoke):
        completion = json.loads((smoke/'completed.json').read_text())
        record = json.loads((smoke/'run_record.json').read_text())
        condition = json.loads((smoke/'condition_verified.json').read_text())
        files = sorted((smoke/'csv').glob('version_*/metrics.csv'))
        rows = list(csv.DictReader(files[-1].open()))
        losses = [float(r['train/loss_step']) for r in rows if r.get('train/loss_step')]
        if completion['global_step'] != 2 or record['global_batch'] != 256 or not losses or not all(map(math.isfinite, losses)):
            raise RuntimeError('Full-size RECAP smoke verification failed')
        self.record(smoke_verification=dict(passed=True, global_batch=256, local_sft=16,
            local_rollout=16, losses=losses, condition=condition))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--value-epochs', type=int, default=3)
    parser.add_argument('--stall-seconds', type=int, default=1800)
    parser.add_argument('--profile', choices=['legacy','rlinf_1563'], default='legacy')
    pipeline = Pipeline(parser.parse_args())
    import fcntl
    lock = (pipeline.root/'pipeline.lock').open('w')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('The RECAP pipeline already has a running controller')
    try:
        pipeline.run_forever()
    except Exception as error:
        pipeline.record(state='failed', error=f'{type(error).__name__}: {error}')
        raise
