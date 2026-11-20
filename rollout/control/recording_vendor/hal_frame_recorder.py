"""Background HAL frame recorder: every ``stride``-th aligned HAL observation of an episode.

``stride`` may be a callable (read for every frame), e.g. 4 while the policy plays at 7.5 Hz and 2 at
15 Hz, so one dataset frame is always one policy step. ``pause`` / ``resume`` skip frames inside an
episode (arm parked); ``freeze`` ends the episode's frames immediately (verdict pressed).

Independent of the control loop (which only samples at waypoints): it follows the HAL observation
stream (30 Hz) and keeps state30 + the three RGB frames (JPEG in memory) of each kept sample.

Each kept frame is labelled by ``label_fn`` (POLICY / HUMAN). Once ``freeze_fn`` returns True
(operator pressed success / failure / home), nothing more is kept until the next episode, so the
frames after the verdict are not part of the episode.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot_v21_writer import VIDEO_KEYS, encode_jpeg

logger = logging.getLogger(__name__)


class HalFrameRecorder:
    def __init__(self, observations: Any, make_observation: Callable[[dict, str], dict], task: str, *,
                 stride: int | Callable[[], int], label_fn: Callable[[], int],
                 freeze_fn: Callable[[], bool] = lambda: False,
                 jpeg_quality: int = 95, max_frames: int = 6000):
        if not callable(stride) and int(stride) < 1:
            raise ValueError("stride must be >= 1")
        self._source = observations  # scripts.deeptouch.hal.ObservationSource
        self._make_observation = make_observation
        self._task = task
        self._stride = stride if callable(stride) else (lambda value=int(stride): value)
        self._label_fn = label_fn
        self._freeze_fn = freeze_fn
        self._jpeg_quality = int(jpeg_quality)
        self._max_frames = int(max_frames)
        self._lock = threading.Lock()
        self._recording = False
        self._frozen = False
        self._paused = False
        self._reset_buffers()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="steam-frame-recorder", daemon=True)
        self._thread.start()

    def _reset_buffers(self) -> None:
        self._states: list[np.ndarray] = []
        self._jpegs: dict[str, list[bytes]] = {key: [] for key in VIDEO_KEYS.values()}
        self._sources: list[int] = []
        self._sequences: list[int] = []
        self._stamps_ns: list[int] = []
        self._last_kept: int | None = None
        self._gaps = 0
        self._bad_frames = 0
        self._capture_sec = 0.0

    # ---- episode control (control thread) ----
    def start_episode(self) -> None:
        with self._lock:
            self._reset_buffers()
            self._frozen = False
            self._paused = False
            self._recording = True
        logger.info("STEAM recorder: episode started (stride %s)", self._stride())

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._last_kept = None  # no gap is counted across a pause

    def freeze(self) -> None:
        if self._recording and not self._frozen:
            self._frozen = True
            logger.info("STEAM recorder: verdict, episode frames frozen at %s", len(self._states))

    @property
    def recording(self) -> bool:
        return self._recording

    def frame_count(self) -> int:
        return len(self._states)

    def finish_episode(self) -> dict[str, Any]:
        """Stop and hand over the buffers (and capture diagnostics)."""
        with self._lock:
            self._recording = False
            out = {
                "states": np.asarray(self._states, dtype=np.float32).reshape(-1, 30),
                "jpeg_frames": self._jpegs,
                "sources": np.asarray(self._sources, dtype=np.int64),
                "observation_sequence": np.asarray(self._sequences, dtype=np.int64),
                "hal_time_ns": np.asarray(self._stamps_ns, dtype=np.int64),
                "sequence_gaps": int(self._gaps),
                "bad_frames": int(self._bad_frames),
                "mean_capture_ms": 1e3 * self._capture_sec / max(1, len(self._states)),
            }
            self._reset_buffers()
        return out

    def abort_episode(self, reason: str) -> None:
        with self._lock:
            was = self._recording
            self._recording = False
            n = len(self._states)
            self._reset_buffers()
        if was:
            logger.warning("STEAM recorder: dropped %s frames (%s)", n, reason)

    def close(self) -> None:
        self._stop.set()
        with self._source.condition:
            self._source.condition.notify_all()
        self._thread.join(timeout=2.0)

    # ---- capture thread ----
    def _next_sample(self, last_sequence: Any) -> tuple[dict, float, Any] | None:
        source = self._source
        with source.condition:
            source.condition.wait_for(
                lambda: self._stop.is_set() or source.error is not None or (
                    source.latest_value is not None and source.latest_value[2] != last_sequence),
                timeout=0.5)
            if self._stop.is_set() or source.error is not None or source.latest_value is None:
                return None
            value = source.latest_value
        return None if value[2] == last_sequence else value

    def _run(self) -> None:
        last_sequence: Any = None
        counter = 0  # fallback when HAL gives no sequence numbers
        while not self._stop.is_set():
            sample = self._next_sample(last_sequence)
            if sample is None:
                if self._source.error is not None:
                    time.sleep(0.1)
                continue
            state, stamp, sequence = sample
            last_sequence = sequence
            counter += 1
            seq = int(sequence) if sequence is not None else counter
            if not self._recording or self._frozen or self._paused:
                continue
            stride = int(self._stride())
            if self._last_kept is not None and seq - self._last_kept < stride:
                continue
            if self._freeze_fn():
                self._frozen = True
                logger.info("STEAM recorder: verdict pressed, episode frames frozen at %s", len(self._states))
                continue
            if len(self._states) >= self._max_frames:
                self._frozen = True
                logger.warning("STEAM recorder: max_frames=%s reached, later frames not recorded", self._max_frames)
                continue
            started = time.monotonic()
            label = int(self._label_fn())
            try:
                observation = self._make_observation(dict(state), self._task)
                jpegs = {VIDEO_KEYS[key]: encode_jpeg(observation["image"][key], self._jpeg_quality)
                         for key in VIDEO_KEYS}
                state30 = np.asarray(observation["state"], dtype=np.float32).reshape(30)
            except Exception as exc:  # one bad frame must not stop the rollout
                self._bad_frames += 1
                logger.warning("STEAM recorder: skipped frame seq=%s: %s", seq, exc)
                continue
            with self._lock:
                if not self._recording or self._frozen or self._paused:
                    continue
                if self._last_kept is not None and seq - self._last_kept > stride:
                    self._gaps += 1
                self._last_kept = seq
                self._states.append(state30)
                for key, data in jpegs.items():
                    self._jpegs[key].append(data)
                self._sources.append(label)
                self._sequences.append(seq)
                self._stamps_ns.append(int(round(float(stamp) * 1e9)))
                self._capture_sec += time.monotonic() - started


def trim_idle_runs(frames: dict[str, Any], label: int, *, min_move_m: float = 0.001,
                   min_gripper_m: float = 0.001, keep: int = 1) -> tuple[dict[str, Any], int]:
    """Drop the still frames at both ends of every ``label`` run (e.g. the operator getting ready).

    A frame is "moving" once the left EEF xyz (state[7:10]) or gripper (state[14]) differs from the
    run's first (leading side) / last (trailing side) frame by more than the thresholds; ``keep`` still
    frames stay as context. A run that never moves is dropped entirely. Returns (frames, dropped).
    """
    sources = np.asarray(frames["sources"])
    states = np.asarray(frames["states"])
    mask = np.ones(len(sources), dtype=bool)
    index = 0
    while index < len(sources):
        if sources[index] != label:
            index += 1
            continue
        end = index
        while end < len(sources) and sources[end] == label:
            end += 1
        xyz, grip = states[index:end, 7:10], states[index:end, 14]

        def moved(ref: int) -> np.ndarray:
            return ((np.linalg.norm(xyz - xyz[ref], axis=1) > min_move_m)
                    | (np.abs(grip - grip[ref]) > min_gripper_m))

        lead, tail = moved(0), moved(-1)
        if not lead.any():
            mask[index:end] = False
        else:
            first = int(np.argmax(lead))
            last = int(len(tail) - 1 - np.argmax(tail[::-1]))
            mask[index:index + max(0, first - keep)] = False
            mask[index + min(end - index, last + 1 + keep):end] = False
        index = end
    if mask.all():
        return frames, 0
    out = dict(frames)
    for key in ("states", "sources", "observation_sequence", "hal_time_ns"):
        out[key] = np.asarray(frames[key])[mask]
    out["jpeg_frames"] = {key: [f for f, m in zip(values, mask) if m] for key, values in frames["jpeg_frames"].items()}
    return out, int((~mask).sum())
