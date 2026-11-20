import logging
import threading
from types import SimpleNamespace
import unittest

from control_timing import BackgroundLogging, install_diagnostics


class TimingTests(unittest.TestCase):
    def test_blocked_log_sink_does_not_block_control_and_flushes(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        records = []

        class Sink(logging.Handler):
            def emit(self, record):
                entered.set()
                release.wait(2)
                records.append(record.getMessage())

        logger = logging.Logger('timing-test', logging.INFO)
        logger.addHandler(Sink())
        session = BackgroundLogging(logger)

        def control():
            logger.info('request %s', 7)
            returned.set()

        thread = threading.Thread(target=control)
        try:
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertTrue(returned.wait(.05), 'control blocked on log output')
        finally:
            release.set()
            thread.join(2)
            session.close()
        self.assertEqual(records, ['request 7'])

    def test_diagnostics_preserves_validation_errors_and_timeout(self):
        class Window:
            def receive(self, timeout):
                self.timeout = timeout
                raise TimeoutError('missing result')

        def validate(request, chunk):
            raise ValueError('conditioned prefix mismatch')

        stable = SimpleNamespace(validate_prefix_handover=validate)
        runner = SimpleNamespace(GripperExecutionWindow=Window,
                                 install_crop_pin=lambda *args: None,
                                 LOG=logging.getLogger('timing-test'))
        close = install_diagnostics(runner)
        try:
            runner.install_crop_pin(stable, {})
            with self.assertRaisesRegex(ValueError, 'prefix mismatch'):
                stable.validate_prefix_handover(SimpleNamespace(serial=7), {})
            window = Window()
            with self.assertRaisesRegex(TimeoutError, 'missing result'):
                window.receive(.05)
            self.assertEqual(window.timeout, .05)
        finally:
            close()


if __name__ == '__main__':
    unittest.main()
