"""Log received VR actions without altering button mappings or robot commands."""
import json
import logging
import sys
import time

from common import ROOT, SOURCE


def install():
    sys.path.insert(0, str(SOURCE.parent / 'vr'))
    import deeptouch_vr_controls as controls
    original = controls.process_vr_snapshot
    last_write = [0.0]
    last_event = [None]

    def observe(snap, *, mapper, last_ref):
        actions, ref = original(snap, mapper=mapper, last_ref=last_ref)
        now = time.monotonic()
        if actions:
            last_event[0] = {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                             'actions': [name for name, _ in actions]}
            logging.getLogger('recap.vr').info('VR input epoch=%s sequence=%s actions=%s',
                                              snap.epoch, snap.sequence, last_event[0]['actions'])
        if actions or now - last_write[0] >= 1.0:
            frame = getattr(snap, 'frame', None)
            received = getattr(snap, 'received_at', None)
            state = {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                     'connection': str(getattr(snap, 'connection_state', 'unknown')),
                     'epoch': getattr(snap, 'epoch', None),
                     'sequence': getattr(snap, 'sequence', None),
                     'age_s': None if received is None else (time.monotonic_ns() - received) / 1e9,
                     'left_y': None if frame is None else int(frame.left_keys.two),
                     'last_event': last_event[0]}
            path = ROOT / 'runs/vr_input_status.json'
            try:
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(state))
                temporary.replace(path)
            except OSError:
                logging.getLogger('recap.vr').exception('Unable to write VR input status')
            last_write[0] = now
        return actions, ref

    controls.process_vr_snapshot = observe
