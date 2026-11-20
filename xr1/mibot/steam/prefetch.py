"""Parallel pair decoding with the upstream sampler and resumable consumed RNG.

Sampling stays sequential. Only deterministic image decoding/collation runs ahead;
checkpoints capture the RNG after the last consumed batch, never the queued ones.
"""

import copy
import multiprocessing
import queue
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from .upstream.data.pair_dataset import PairDataset


def _init_batch_worker(language_path, cameras, num_bins, decode_workers):
    from .upstream.data.pair_dataset import BinaryPairDataCollator
    from .upstream.models.steam.processing import SteamProcessor

    global _worker_collate, _worker_decoders
    # Keep the parallel CPU interpolation kernel used by the original trainer.
    # Its uint8 rounding can differ by one level from the single-thread kernel.
    torch.set_num_threads(2)
    _worker_collate = BinaryPairDataCollator(
        SteamProcessor(
            tokenizer_name_or_path=language_path, image_keys=cameras, do_augment=False
        ),
        num_bins=num_bins,
        train=False,
    )
    _worker_decoders = ThreadPoolExecutor(max_workers=decode_workers)


def _decode_packed_request(packed):
    from .data import XR1TrajectorySource

    record, cameras, source_name, kwargs = packed
    source = XR1TrajectorySource([record])
    dataset = object.__new__(PairDataset)
    dataset._source, dataset.camera_keys, dataset.source_name = (
        source,
        cameras,
        source_name,
    )
    kwargs = kwargs.copy()
    kwargs["raw_t"], kwargs["raw_tk"] = source.load_pair_views(
        0, kwargs["frame_idx_t"], kwargs["frame_idx_tk"], cameras
    )
    return PairDataset._build_sample(dataset, **kwargs)


def _process_batch(requests):
    return _worker_collate(list(_worker_decoders.map(_decode_packed_request, requests)))


@dataclass
class PairRequest:
    dataset: PairDataset
    kwargs: dict

    def pack(self):
        return (
            self.dataset.source.records[self.kwargs["episode"]],
            self.dataset.camera_keys,
            self.dataset.source_name,
            self.kwargs,
        )

    def load(self):
        kwargs = self.kwargs.copy()
        source = self.dataset.source
        if hasattr(source, "load_pair_views"):
            kwargs["raw_t"], kwargs["raw_tk"] = source.load_pair_views(
                kwargs["episode"],
                kwargs["frame_idx_t"],
                kwargs["frame_idx_tk"],
                self.dataset.camera_keys,
            )
        return PairDataset._build_sample(self.dataset, **kwargs)


class DeferredPairDataset(PairDataset):
    def _build_sample(self, **kwargs):
        return PairRequest(self, kwargs)


def pin(value):
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, dict):
        return {k: pin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(pin(v) for v in value)
    return value


class PairPrefetcher:
    def __init__(
        self,
        groups,
        probabilities,
        rng,
        batch,
        collate,
        workers=12,
        depth=2,
        pin_memory=True,
        processes=0,
        language_path=None,
        num_bins=32,
    ):
        self.groups, self.probabilities = groups, probabilities
        self.rng = rng
        self.ahead_rng = np.random.default_rng()
        self.ahead_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        for ds in groups:
            ds._rng = self.ahead_rng
        self.batch, self.collate = batch, collate
        self.workers, self.pin_memory = workers, pin_memory
        self.processes, self.language_path, self.num_bins = (
            processes,
            language_path,
            num_bins,
        )
        self.queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            if self.processes:
                self._produce_processes()
                return
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                while not self.stop.is_set():
                    requests, state = self._sample()
                    items = list(pool.map(lambda request: request.load(), requests))
                    data = self.collate(items)
                    self._put((pin(data) if self.pin_memory else data, state))
        except BaseException as exc:  # noqa: BLE001 - deliver worker failures to the consumer
            self._put(exc)

    def _sample(self):
        requests = []
        for _ in range(self.batch):
            ds = self.groups[
                int(self.ahead_rng.choice(len(self.groups), p=self.probabilities))
            ]
            requests.append(ds[int(self.ahead_rng.integers(len(ds)))])
        return requests, copy.deepcopy(self.ahead_rng.bit_generator.state)

    def _produce_processes(self):
        cameras = self.groups[0].camera_keys
        with ProcessPoolExecutor(
            max_workers=self.processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_batch_worker,
            initargs=(self.language_path, cameras, self.num_bins, self.workers),
        ) as pool:
            pending = deque()

            def submit():
                requests, state = self._sample()
                pending.append(
                    (pool.submit(_process_batch, [r.pack() for r in requests]), state)
                )

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
