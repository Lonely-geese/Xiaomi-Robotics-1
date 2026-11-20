"""Ordered CPU policy batches with checkpointing at the consumed sampler state."""

import copy
import multiprocessing
import os
import queue
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

from .prefetch import pin


def prepare_batch(dataset, collate, indices, seed, decoders=None):
    requests = [(int(index), seed + i) for i, index in enumerate(indices)]
    items = list(
        map(dataset.load_seeded, requests)
        if decoders is None
        else decoders.map(dataset.load_seeded, requests)
    )
    generator = torch.Generator().manual_seed(seed)
    return collate(items, random_values=torch.rand(len(items), generator=generator))


def _init_worker(config, decode_workers, cpu_threads):
    import faulthandler
    import signal
    import av
    from .data import PolicyCollator, PolicyDataset

    global _dataset, _collate, _decoders
    faulthandler.register(signal.SIGUSR1)
    # FFmpeg's frame-thread logging callbacks can deadlock with concurrent PyAV
    # decoders. Parallelize independent samples, using one decoder thread each.
    av.logging.set_level(None)
    os.environ["XR1_VIDEO_THREADS"] = "1"
    os.environ["XR1_VIDEO_THREAD_TYPE"] = "SLICE"
    torch.set_num_threads(cpu_threads)
    _dataset = PolicyDataset(
        config["manifest"], config["advantages"],
        stride=config.get("sample_stride", 1),
        allow_smoke=config.get("smoke_only", False),
    )
    _collate = PolicyCollator(config["actor_vlm_path"], config.get("unconditional_prob", 0.1))
    _decoders = ThreadPoolExecutor(max_workers=decode_workers)


def _process_batch(indices, seed):
    return prepare_batch(_dataset, _collate, indices, seed, _decoders)


class PolicyPrefetcher:
    def __init__(self, config, samples, rng, batch, *, processes=4,
                 decode_workers=4, depth=3, cpu_threads=2, pin_memory=True):
        if min(processes, decode_workers, depth, cpu_threads) < 1:
            raise ValueError("prefetch worker counts and depth must be positive")
        self.config, self.samples, self.batch = config, samples, batch
        self.rng = rng
        self.ahead_rng = np.random.default_rng()
        self.ahead_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        self.processes, self.decode_workers = processes, decode_workers
        self.cpu_threads, self.pin_memory = cpu_threads, pin_memory
        self.queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _sample(self):
        indices = self.ahead_rng.integers(self.samples, size=self.batch)
        seed = int(self.ahead_rng.integers(0, 2**63 - self.batch))
        return indices, seed, copy.deepcopy(self.ahead_rng.bit_generator.state)

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            if self.pin_memory:
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            with ProcessPoolExecutor(
                max_workers=self.processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.config, self.decode_workers, self.cpu_threads),
            ) as pool:
                pending = deque()

                def submit():
                    indices, seed, state = self._sample()
                    pending.append((pool.submit(_process_batch, indices, seed), state))

                for _ in range(self.processes):
                    submit()
                while not self.stop.is_set():
                    future, state = pending.popleft()
                    data = future.result()
                    self._put((pin(data) if self.pin_memory else data, state))
                    if not self.stop.is_set():
                        submit()
                for future, _ in pending:
                    future.cancel()
        except BaseException as exc:
            self._put(exc)

    def __next__(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        data, state = item
        self.rng.bit_generator.state = state
        return data

    def close(self):
        self.stop.set()
        self.thread.join()
