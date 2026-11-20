"""Keep log I/O off the servo thread and diagnose handover stalls."""
from collections import deque
import copy
import gc
import logging
from logging.handlers import QueueHandler, QueueListener
import queue
import time


class DeferredLogHandler(QueueHandler):
    def prepare(self, record):
        # Formatting and exception rendering also belong on the logging thread.
        return copy.copy(record)


class BackgroundLogging:
    def __init__(self, logger=None):
        self.logger = logging.getLogger() if logger is None else logger
        if logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        self.handlers = list(self.logger.handlers)
        self.queue = queue.SimpleQueue()
        self.handler = DeferredLogHandler(self.queue)
        self.listener = QueueListener(self.queue, *self.handlers, respect_handler_level=True)
        self.logger.handlers = [self.handler]
        self.listener.start()

    def close(self):
        self.logger.handlers = self.handlers
        self.listener.stop()


def install_diagnostics(runner):
    """Observe delays without changing deadlines, validation or motion commands."""
    slow_gc = deque(maxlen=16)
    gc_started = {}

    def gc_event(phase, info):
        generation = info['generation']
        now = time.monotonic()
        if phase == 'start':
            gc_started[generation] = now
        else:
            started = gc_started.pop(generation, now)
            if now - started >= .01:
                slow_gc.append((started, now - started, generation))

    gc.callbacks.append(gc_event)
    original_receive = runner.GripperExecutionWindow.receive

    def receive(self, timeout):
        started = time.monotonic()
        try:
            return original_receive(self, timeout)
        finally:
            elapsed = time.monotonic() - started
            if elapsed > max(0, timeout) + .01:
                runner.LOG.warning('Handover receive/window took %.6fs (wait budget %.6fs); recent slow GC=%s',
                                   elapsed, timeout, list(slow_gc))

    runner.GripperExecutionWindow.receive = receive
    original_pin = runner.install_crop_pin

    def install_pin(stable, identity):
        original_pin(stable, identity)
        validate = stable.validate_prefix_handover

        def timed_validate(request, chunk):
            started = time.monotonic()
            try:
                return validate(request, chunk)
            finally:
                elapsed = time.monotonic() - started
                if elapsed >= .01:
                    runner.LOG.warning('Handover prefix validation request %s took %.6fs; recent slow GC=%s',
                                       request.serial, elapsed, list(slow_gc))

        stable.validate_prefix_handover = timed_validate

    runner.install_crop_pin = install_pin

    def close():
        gc.callbacks.remove(gc_event)

    return close
