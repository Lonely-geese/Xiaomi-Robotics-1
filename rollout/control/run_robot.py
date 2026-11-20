"""VR episode control with the RECAP model, norm and camera arrangement pinned."""
import os
import sys
import argparse
from pathlib import Path

from common import (ROOT, SOURCE, RESET_CONFIG, check_handshake, controller_conflicts,
                    load, verify_dependencies, verify_package)
from gripper_snap import install_closing_gripper_snap


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--record-out', type=Path)
    options, argv = parser.parse_known_args()
    read_only = any(arg in argv for arg in ('--dry-run', '--probe', '--observe-infer-test', '--help', '-h'))
    if not read_only:
        conflicts = controller_conflicts()
        if conflicts:
            raise SystemExit('Stop existing robot controllers before starting RECAP: ' + str(conflicts))
    identity = verify_package(require_weights=not read_only)
    verify_dependencies()
    os.environ['XR1_RESET_CONFIG'] = RESET_CONFIG
    from vr_observer import install as install_vr_observer
    install_vr_observer()
    runner = load('usb_recap_vr_runtime', SOURCE)
    runner.HERE = ROOT
    runner.VR_CONTROLS = ROOT.parent / 'runtime/vr'
    original_pin = runner.install_crop_pin

    def install_pin(stable, selected):
        if selected != identity:
            raise RuntimeError('RECAP controller requires its pinned model manifest')
        original_pin(stable, selected)
        base = stable.PolicyClient

        class Client(base):
            def __init__(self, *args, **kwargs):
                try:
                    super().__init__(*args, **kwargs)
                    check_handshake(self.metadata, identity)
                except Exception:
                    if hasattr(self, 'socket'):
                        self.close()
                    raise

        stable.PolicyClient = Client

    runner.install_crop_pin = install_pin
    runner.install_gripper_snap = install_closing_gripper_snap
    from manual_gripper import install as install_manual_gripper
    install_manual_gripper(runner)
    from control_timing import BackgroundLogging, install_diagnostics
    background_logging = BackgroundLogging()
    close_diagnostics = install_diagnostics(runner)
    recording = None
    try:
        if options.record_out is not None and not read_only:
            from recording import enable_recording
            recording = enable_recording(runner, options.record_out, identity)
        runner.main(['--manifest', str(ROOT / 'model_manifest.json'), '--server-port', '10094',
                     '--async-control-hz', '7.5', *argv])
    finally:
        try:
            if recording is not None:
                recording.close()
        finally:
            close_diagnostics()
            background_logging.close()


if __name__ == '__main__':
    main()
