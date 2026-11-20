"""Process-local stop compatibility; does not change the stable motion pipeline."""
import time


def install(stable, helper, args):
    class HAL(stable.CartesianHAL):
        def connect(self):
            super().connect()
            self.stop_backend = helper.select_stop_backend(self.client.get_capabilities())
            self.stop_limits = helper.StopLimits(
                timeout=args.settle_timeout, stable_s=max(.2, args.settle_stable_s),
                feedback_age=args.max_feedback_age,
                linear_speed=args.settle_linear_speed_m_s,
                angular_speed=args.settle_angular_speed_rad_s,
                joint_speed=args.settle_joint_speed_deg_s,
                position_drift=args.max_follow_error_m,
                rotation_drift=args.max_follow_error_rad)
            stable.LOG.info('HAL stop backend: %s (fresh measured rest required)', self.stop_backend)

        @staticmethod
        def assert_mode(mode):
            if mode.mode != stable.MotionMode.TRAJECTORY or mode.command_space != stable.CommandSpace.CART:
                raise stable.HardwareError('HAL has not acknowledged TRAJECTORY/CART')

        def stop_tracking(self):
            if not self.tracking:
                return False
            anchor, *_ = self.feedback_with_metadata()
            if self.stop_backend == 'native_stop_operation':
                result = self.client.execute_op('stop_online_tracking', {'unit': 'leftarm'},
                                                wait=True, timeout_ms=10000)
                payload = result.get('result') or {}
                if (not result.get('success') or payload.get('unit') != 'leftarm'
                        or payload.get('mode') != 'TRAJECTORY'
                        or payload.get('command_space') != 'CART'
                        or payload.get('soft_stopping') is not False):
                    raise stable.HardwareError(f'Invalid native soft-stop result: {result}')
            elif self.stop_backend == 'mode_switch_feedback':
                self.assert_mode(self.client.set_motion_mode(
                    stable.MotionMode.TRAJECTORY, stable.CommandSpace.CART, units=['leftarm']))
            else:
                raise stable.HardwareError('No negotiated stop backend')
            # Start after the ACK: neither old cached samples nor ACK alone imply rest.
            checker = helper.ArmStopped(anchor, time.monotonic(), self.stop_limits)
            deadline = time.monotonic() + self.stop_limits.timeout
            while time.monotonic() < deadline:
                pose, joints, _, _, stamp, sequence, _ = self.feedback_with_metadata()
                status = checker.update(pose, joints, stamp, sequence, time.monotonic())
                if status['confirmed']:
                    self.assert_mode(self.client.get_motion_mode())
                    self.tracking = False
                    stable.LOG.info('Left arm measured stop confirmed: %s', status)
                    return True
                time.sleep(1 / self.feedback_hz)
            raise stable.HardwareError('Soft stop not verified; do not touch the robot')

    class Source(stable.ObservationSource):
        def close(self):
            # Clean up even if connection fails before the reader thread starts.
            if self.thread.ident is None:
                self.stop.set()
                try:
                    if self.stream is not None:
                        self.stream.close()
                finally:
                    self.client.close()
                return
            return super().close()

    stable.CartesianHAL = HAL
    stable.ObservationSource = Source
