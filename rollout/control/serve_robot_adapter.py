"""RECAP native TCP inference -> existing Cartesian robot protocol."""
import argparse
import logging
from pathlib import Path
import signal
import sys
import threading

import numpy as np
from PIL import Image

from common import (ROOT, PACKAGE, MODEL, NATIVE_CODE, TASK, EXTENDED_IDENTITY,
                    verify_dependencies, verify_package, verify_native)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=10094)
    parser.add_argument('--native-port', type=int, default=10090)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    identity = verify_package(full=True)
    profile = verify_dependencies()
    native_pid = verify_native(args.native_port)
    # Import the client before the stable adapter adds its legacy code directory.
    sys.path.insert(0, str(NATIVE_CODE))
    from usb_client import USBClient
    import mibot.server.runtime.client as native_client
    if not Path(native_client.__file__).resolve().is_relative_to(NATIVE_CODE):
        raise RuntimeError('Wrong native client code imported')
    adapter = profile.load('usb_recap_cartesian_adapter', profile.STABLE / 'serve_adapter.py')
    adapter.ROOT = ROOT

    class Runtime:
        model_dir = MODEL

        def __init__(self):
            self.client = USBClient('127.0.0.1', args.native_port)
            self.client.socket.settimeout(30)
            self.failed = False

        def predict(self, raw, action_prefix=None):
            if self.failed:
                raise RuntimeError('Native connection failed; restart the RECAP adapter')
            if raw.get('task') != TASK:
                raise ValueError('This model requires the USB task prompt')
            images = []
            for key in ('head_rgb', 'left_wrist_rgb'):
                value = np.asarray(raw['images'][key])
                if value.shape != (3, 480, 848) or value.dtype != np.uint8:
                    raise ValueError(f'{key}: expected original uint8 RGB (3,480,848)')
                images.append(Image.fromarray(value.transpose(1, 2, 0)))
            try:
                return self.client.infer(adapter.robot_state(raw), *images,
                                         action_prefix=action_prefix)
            except Exception:
                self.failed = True
                self.client.close()
                raise

        def close(self):
            self.client.close()

    class RecapPolicy(adapter.Policy):
        def __init__(self, runtime):
            # The frozen Policy constructor only accepts literal-dict configs;
            # this package uses dict(...). verify_package validates it above.
            # Reuse its inference, prefix checking, Cartesian limits and logging.
            self.runtime = runtime
            self.limits = adapter.CartesianLimits()
            self.lock = threading.Lock()
            self.prediction_writer = adapter.PredictionWriter(ROOT / 'runs/predictions')
            self.warmed = False
            self.handshake = {
                'status': 'ready', 'adapter': 'xr1_official_prefix_cartesian_v1',
                **{key: identity[key] for key in ('checkpoint_step', 'checkpoint_sha256',
                                                'checkpoint_epoch', 'completed_epochs')},
                'model_profile': identity['profile'],
                'action_horizon': 30, 'frequency_hz': 30, 'active_arm': 'left',
                'action_space': 'world_cartesian_xyz_xyzw', 'async_prefix': True,
                'max_prefix_length': 6, 'wire_format': adapter.WIRE_FORMAT,
                'trajectory_rejection_protocol': 'structured_retry_v1',
                **{key: identity[key] for key in EXTENDED_IDENTITY},
                'native_pid': native_pid,
            }

    runtime = Runtime()
    policy = server = None
    try:
        policy = RecapPolicy(runtime)
        server = adapter.ThreadedServer(('127.0.0.1', args.port), adapter.Handler)
        server.policy = policy
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        logging.info('RECAP Cartesian adapter ready at 127.0.0.1:%s; native PID %s; norm=%s',
                     args.port, native_pid, MODEL / 'normalize.json')
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        if policy is not None:
            policy.close()
        runtime.close()


if __name__ == '__main__':
    main()
