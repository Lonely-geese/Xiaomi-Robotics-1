import logging
import math
from types import SimpleNamespace
import unittest

from gripper_snap import install_closing_gripper_snap


class FakeHAL:
    def _validate_grippers(self, left, right=None):
        if not math.isfinite(left) or not -1e-6 <= left <= 0.09 + 1e-6:
            raise ValueError('invalid gripper target')
        return min(0.09, max(0.0, float(left))), right

    def feedback(self):
        return None, None, 0.09, 0.09

    def set_grippers(self, left, right=None):
        return self._validate_grippers(left, right)

    def send_command(self, pose, left_gripper=None):
        if getattr(self, 'fail_send', False):
            raise RuntimeError('send failed')
        return pose, left_gripper


class ClosingSnapTests(unittest.TestCase):
    def setUp(self):
        stable = SimpleNamespace(CartesianHAL=FakeHAL, LOG=logging.getLogger(__name__))
        install_closing_gripper_snap(stable, 0.008)
        self.hal = stable.CartesianHAL()
        self.hal.set_grippers(0.09)

    def commands(self, targets):
        return [self.hal.send_command(None, target)[1] for target in targets]

    def test_close_hold_then_confirm_release(self):
        self.assertEqual(
            self.commands([0.02, 0.008, 0.006, 0.006, 0.002, 0.003, 0.004, 0.009, 0.01, 0.012]),
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012])
        # Once released, the rest of this episode bypasses the close threshold.
        self.assertEqual(self.commands([0.004, 0.003, 0.007, 0.02]), [0.004, 0.003, 0.007, 0.02])

    def test_recorded_grasp_noise_does_not_release(self):
        # Actual raw targets at serial 5, indices 20/21/22/27/29 on 2026-11-05.
        targets = [0.003848165273666382, 0.0033256933093070984,
                   0.0034022778272628784, 0.003153197467327118,
                   0.0037270858883857727, 0.0031472714617848396,
                   0.0031377356499433517, 0.003101782640442252,
                   0.0031173797324299812]
        self.assertEqual(self.commands(targets), [0.0] * len(targets))

    def test_release_confirmation_must_be_consecutive(self):
        self.assertEqual(self.commands([0.005, 0.009, 0.01, 0.008, 0.009, 0.01]), [0.0] * 6)
        self.assertEqual(self.commands([0.011]), [0.011])

    def test_reset_manual_and_pose_only_commands(self):
        self.commands([0.005])
        self.assertEqual(self.hal.send_command(None), (None, None))
        self.assertEqual(self.commands([0.005]), [0.0])
        self.assertEqual(self.commands([0.009, 0.01, 0.012]), [0.0, 0.0, 0.012])
        self.hal.set_grippers(0.09)
        self.assertEqual(self.commands([0.008]), [0.0])

    def test_failed_send_does_not_advance_release(self):
        self.commands([0.005, 0.009, 0.01])
        self.hal.fail_send = True
        with self.assertRaises(RuntimeError):
            self.hal.send_command(None, 0.012)
        self.assertEqual(self.hal._snap_phase, 'holding')
        self.assertEqual(self.hal._snap_release_count, 2)
        self.hal.fail_send = False
        self.assertEqual(self.commands([0.004]), [0.0])

    def test_invalid_targets_are_not_hidden(self):
        for value in [-0.001, float('nan'), float('inf'), 0.1]:
            with self.assertRaises(ValueError):
                self.hal.send_command(None, value)


if __name__ == '__main__':
    unittest.main()
