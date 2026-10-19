#!/usr/bin/env python3
"""Show training loss and observed ETA without changing the training process."""
import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def report(run_dir, progress):
    latest_loss = None
    loss_step = None
    for path in run_dir.glob('csv/version_*/metrics.csv'):
        with path.open() as source:
            for row in csv.DictReader(source):
                if row.get('train/loss_step') and row.get('step'):
                    step = int(row['step']) + 1
                    if step <= progress['global_step'] and (loss_step is None or step > loss_step):
                        loss_step, latest_loss = step, float(row['train/loss_step'])
    # The training callback writes progress just before Lightning flushes metrics.
    # Wait for the loss at this update so a report never labels an older loss as current.
    if loss_step != progress['global_step']:
        return None
    seconds = progress.get('estimated_training_seconds_remaining')
    finish = (datetime.fromtimestamp(progress['updated_unix'], ZoneInfo('Asia/Shanghai'))
        + timedelta(seconds=seconds)).isoformat() if seconds is not None else None
    record = dict(**progress, loss=latest_loss, loss_step=loss_step,
        estimated_training_days_remaining=seconds / 86400 if seconds is not None else None,
        estimated_training_finish_beijing=finish,
        estimate_note='Observed training speed; validation, checkpoints and interruptions add time.')
    loss_text = f'{latest_loss:.4f}' if latest_loss is not None else 'pending'
    speed = progress.get('seconds_per_update')
    speed_text = f'{speed:.2f}s/update' if speed is not None else 'pending'
    remaining = f'{seconds / 86400:.2f} days' if seconds is not None else 'pending'
    print(f"Step {progress['global_step']}/{progress['max_steps']} | loss {loss_text} "
        f"(step {loss_step}) | {speed_text} | training remaining {remaining} | "
        f"finish Beijing {finish}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    assert args.interval > 0
    queue = Path(__file__).resolve().parents[2] / '.queue/left_relative_10ep'
    last_bucket = None
    with (queue / 'progress_every100.jsonl').open('a') as output:
        while True:
            target = args.run_dir / 'training_progress.json'
            if target.exists():
                progress = json.loads(target.read_text())
                bucket = progress['global_step'] // args.interval
                if last_bucket is None or bucket > last_bucket or args.once:
                    record = report(args.run_dir, progress)
                    if record is None:
                        time.sleep(1)
                        continue
                    output.write(json.dumps(record) + '\n')
                    output.flush()
                    last_bucket = bucket
                if args.once:
                    return
            state = json.loads((queue / 'status.json').read_text())['state']
            if state in ('failed', 'stopped', 'completed'):
                print('Queue state: ' + state, flush=True)
                return
            time.sleep(5)


if __name__ == '__main__':
    main()
