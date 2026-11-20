import threading
import time

class FakeObservationSource:
    """Same attributes the recorder reads from hal.ObservationSource; publishes at 30 Hz."""

    def __init__(self):
        self.condition = threading.Condition()
        self.latest_value = None
        self.error = None
        self.sequence = 1000
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.sequence += 1
            if self.sequence % 50 == 0:  # simulate one dropped HAL sample
                self.sequence += 1
            state = {"seq": self.sequence}
            with self.condition:
                self.latest_value = (state, time.monotonic(), self.sequence)
                self.condition.notify_all()
            time.sleep(1 / 30)

    def close(self):
        self._stop.set()
