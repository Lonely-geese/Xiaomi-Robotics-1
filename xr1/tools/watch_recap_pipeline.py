#!/usr/bin/env python3
"""Keep the RECAP controller alive; do not launch duplicate workers."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['legacy','rlinf_1563','events_1563'], default='legacy')
    parser.add_argument('--stop-after', choices=['critic','policy'], default='policy')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = repo/'.queue'/('recap_rlinf_1563_20261020' if args.profile == 'rlinf_1563' else 'recap_20261020')
    if args.profile == 'events_1563':
        root = repo/'.queue/recap_events_1563_20261021'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'watchdog.lock').open('w')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A RECAP watchdog is already running')
    restarts = 0
    while True:
        if (root/'invalidated.json').exists():
            raise SystemExit(f'Refusing to restart invalidated experiment: {root}')
        with (root/'controller.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(repo/'xr1/tools/run_recap_pipeline.py'),
                '--profile', args.profile, '--stop-after', args.stop_after],
                cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (root/'watchdog_status.json').write_text(json.dumps(dict(watchdog_pid=os.getpid(),
                controller_pid=process.pid, controller_restarts=restarts, state='running'), indent=2))
            code = process.wait()
        try:
            status = json.loads((root/'status.json').read_text())
        except (OSError, ValueError):
            status = {}
        if code == 0 and status.get('stage') == 'complete' and status.get('state') == 'completed':
            (root/'watchdog_status.json').write_text(json.dumps(dict(watchdog_pid=os.getpid(),
                controller_restarts=restarts, state='completed'), indent=2))
            return
        restarts += 1
        print(json.dumps(dict(controller_exit=code, restarting_in_seconds=30, restarts=restarts)), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
