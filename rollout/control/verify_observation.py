"""Read actual cameras/state and check ordinary + six-frame-prefix inference.

No CartesianHAL is constructed and no robot control is acquired.
"""
import json
import os
import time

import numpy as np

from common import ROOT, RESET_CONFIG, check_handshake, verify_dependencies, verify_package


def main():
    identity = verify_package()
    profile = verify_dependencies()
    os.environ['XR1_MODEL_MANIFEST'] = str(ROOT / 'model_manifest.json')
    os.environ['XR1_RESET_CONFIG'] = RESET_CONFIG
    stable = profile.load('recap_observation_check', profile.STABLE / 'run_robot.py')
    source = stable.ObservationSource('127.0.0.1', 7860, 1.0)
    client = None
    try:
        source.start()
        state, stamp, _ = source.latest()
        raw = stable.build_raw(state, 'Pick up the USB cable and insert it into the USB port.')
        client = stable.PolicyClient('127.0.0.1', 10094, 30)
        check_handshake(client.metadata, identity)
        rows = []
        prefix = None
        for serial in (1, 2):
            start = time.monotonic()
            chunk, metadata = client.request(raw, serial, prefix)
            assert chunk['left_pose'].shape == (30, 7)
            assert np.isfinite(chunk['left_pose']).all()
            assert np.isfinite(chunk['raw_action']).all()
            if prefix is not None:
                np.testing.assert_allclose(chunk['raw_action'][:6, :7], prefix[:, :7], atol=1e-3, rtol=0)
            rows.append({'serial': serial, 'prefix_length': 0 if prefix is None else 6,
                         'latency_s': round(time.monotonic() - start, 4),
                         'shape': list(chunk['left_pose'].shape)})
            prefix = chunk['raw_action'][:6].copy()
            print(json.dumps(rows[-1]), flush=True)
        report = {'profile': identity['profile'], 'checkpoint_sha256': identity['checkpoint_sha256'],
                  'normalization_sha256': identity['normalization_sha256'],
                  'recap_conditioning': 'language', 'recap_guidance_scale': 1.0,
                  'real_observation_inference': True, 'action_prefix_6_verified': True,
                  'control_acquired': False, 'policy_commands_sent': False,
                  'results': rows, 'verified_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        (ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        if client is not None:
            client.close()
        source.close()


if __name__ == '__main__':
    main()
