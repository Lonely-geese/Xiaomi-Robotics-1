"""Latch the first grasp until three targets confirm the release phase."""

RELEASE_CONFIRM_TARGETS = 3


def install_closing_gripper_snap(stable, snap_m):
    base = stable.CartesianHAL

    class HAL(base):
        def __init__(self, *args, **kwargs):
            self._snap_phase = 'closing'
            self._snap_release_count = 0
            super().__init__(*args, **kwargs)
            stable.LOG.info(
                'Gripper snap: latch at <= %.4f m; release after %d consecutive targets > %.4f m; '
                'then pass through until reset/manual gripper command',
                snap_m, RELEASE_CONFIRM_TARGETS, snap_m)

        def set_grippers(self, left, right=None):
            result = super().set_grippers(left, right)
            # Reset/manual controls start a fresh grasp cycle. Pausing and
            # reconnecting the inference client do not reset the HAL's latch.
            self._snap_phase = 'closing'
            self._snap_release_count = 0
            return result

        def send_command(self, pose, left_gripper=None):
            if left_gripper is None:
                return super().send_command(pose)
            # Validate the raw target before snapping so invalid commands cannot
            # become valid merely by being below the threshold.
            target, _ = self._validate_grippers(left_gripper)
            phase = self._snap_phase
            release_count = self._snap_release_count
            if phase == 'closing' and target <= snap_m:
                phase = 'holding'
            if phase == 'holding':
                release_count = release_count + 1 if target > snap_m else 0
                if release_count >= RELEASE_CONFIRM_TARGETS:
                    phase = 'released'
            command = 0.0 if phase == 'holding' else target
            result = super().send_command(pose, command)
            if phase != self._snap_phase:
                stable.LOG.info('Gripper phase %s -> %s: raw=%.6f m command=%.6f m',
                                self._snap_phase, phase, target, command)
            self._snap_phase = phase
            self._snap_release_count = release_count
            return result

    stable.CartesianHAL = HAL
