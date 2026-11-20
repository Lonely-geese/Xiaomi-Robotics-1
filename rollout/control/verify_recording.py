"""Validate recording offline or with read-only live HAL frames; never acquire control."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

from common import ROOT, TASK, verify_dependencies
import recording


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'runs/recording_validation' / (('live_' if args.live else 'offline_') + time.strftime('%Y%m%d_%H%M%S'))
    if not args.live:
        path = ROOT / 'test_recording_pipeline.py'
        spec = importlib.util.spec_from_file_location('recorder_pipeline_test', path)
        test = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test)
        test.rx = recording.rx
        test.run(output)
        return
    import numpy as np
    import pyarrow.parquet as pq
    rx = recording.rx
    profile = verify_dependencies()
    stable = profile.load('recap_recording_readonly', profile.STABLE / 'run_robot.py')
    source = stable.ObservationSource('127.0.0.1', 7860, 1.0)
    recorder = None
    writer = rx.LeRobotV21RolloutWriter(output, action_shift=3)
    try:
        source.start()
        source.latest()
        recorder = rx.HalFrameRecorder(source, rx.make_observation_fn(stable), TASK,
                                      stride=1, label_fn=lambda: rx.SOURCE_POLICY,
                                      jpeg_quality=95, max_frames=100)
        recorder.start_episode()
        time.sleep(2)
        frames = recorder.finish_episode()
        assert len(frames['states']) >= 30, len(frames['states'])
        assert np.isfinite(frames['states']).all()
        writer.submit(rx.EpisodeRecord(task='validation_only', states=frames['states'],
            jpeg_frames=frames['jpeg_frames'], sources=frames['sources'],
            observation_sequence=frames['observation_sequence'], hal_time_ns=frames['hal_time_ns'],
            success=False, extra={'validation_only': True, 'robot_commands_sent': 0}))
    finally:
        if recorder is not None:
            recorder.close()
        source.close()
        writer.close()
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    assert table.num_rows >= 30
    videos = list(output.glob('videos/**/*.mp4'))
    assert len(videos) == 3 and all(p.stat().st_size > 0 for p in videos)
    print(json.dumps({'passed': True, 'rows': table.num_rows, 'videos': len(videos),
                      'output': str(output), 'robot_commands_sent': 0}))


if __name__ == '__main__':
    main()
