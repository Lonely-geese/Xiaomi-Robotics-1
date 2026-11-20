"""Offline handover interlock tests. No HAL connections or robot commands."""
import unittest
from types import SimpleNamespace

from recording import install_handover


class HandoverTests(unittest.TestCase):
    def setup_runner(self, events, verify_error=False):
        calls = []
        class Signals:
            def wait(self, accepted, on_idle=None, poll_s=.02):
                calls.append(('wait', accepted))
                event = events.pop(0)
                assert event in accepted
                return event
        runner = SimpleNamespace(VrSignals=Signals, run_policy_segment=lambda *a: ('pause', None))
        def verify(control):
            calls.append(('verify', control))
            if verify_error:
                raise RuntimeError('moving or stale feedback')
        install_handover(runner, None, verify)
        signals = Signals()
        runner.run_policy_segment(None, None, None, 'control', signals)
        return signals, calls

    def test_saved_verdict_verifies_rest_without_second_input(self):
        for verdict in ('success', 'failure'):
            signals, calls = self.setup_runner([verdict])
            self.assertEqual(signals.wait(('pause', 'success', 'failure', 'home')), verdict)
            self.assertEqual(calls, [('wait', ('pause', 'success', 'failure', 'home')),
                                     ('verify', 'control')])

    def test_discard_requires_separate_handback(self):
        signals, calls = self.setup_runner(['home', 'pause'])
        self.assertEqual(signals.wait(('pause', 'success', 'failure', 'home')), 'home')
        self.assertEqual(calls[-2:], [('wait', ('pause',)), ('verify', 'control')])

    def test_resume_verifies_rest(self):
        signals, calls = self.setup_runner(['pause'])
        self.assertEqual(signals.wait(('pause', 'success', 'failure', 'home')), 'pause')
        self.assertEqual(calls[-1], ('verify', 'control'))

    def test_bad_feedback_prevents_resume_or_reset(self):
        for events in (['pause'], ['success'], ['failure'], ['home', 'pause']):
            signals, _ = self.setup_runner(events, verify_error=True)
            with self.assertRaisesRegex(RuntimeError, 'feedback'):
                signals.wait(('pause', 'success', 'failure', 'home'))


if __name__ == '__main__':
    unittest.main()
