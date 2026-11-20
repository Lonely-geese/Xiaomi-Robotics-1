"""Quest USB-local VR buttons → Deeptouch online-RL manual signals (+ headset cues).

Reuses tuozhuai ``VRDevice`` (ADB local → 127.0.0.1:40051). Soft-fails if VR is
unavailable. Keyboard path never calls cue senders.
"""
from __future__ import annotations

import logging
import math
import os
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

LOG = logging.getLogger("deeptouch_vr_controls")

CUE_NEXT = 1
CUE_SUCCESS = 2
CUE_CRITICAL = 3
CUE_FAILURE = 5
CUE_HOME = 9

_CUE_MAGIC = b"DTQC"
_CUE_VERSION = 1
_CUE_HEADER = struct.Struct("<BBHI")  # version, type, length, sequence

# Thumb sectors (teleop binding/thumbstick.py): y-up. Each wedge is 45°.
_THUMB_SECTOR_CENTER_DEG = {
    "right": 0.0,
    "up": 90.0,
    "left": 180.0,
    "down": 270.0,
}
_THUMB_HALF_SECTOR_DEG = 22.5
_THUMB_DEADZONE = 0.35


class _PressState(IntEnum):
    RELEASED = 0
    PRESSED = 1
    LONG_PRESSED = 2


@dataclass
class _SourceState:
    long_mode: bool = False
    state: _PressState = _PressState.RELEASED
    last_state: _PressState = _PressState.RELEASED
    press_start_time: float = 0.0
    long_press_triggered: bool = False


def encode_quest_cue(type_code: int, sequence: int, payload: bytes = b"") -> bytes:
    header = _CUE_HEADER.pack(_CUE_VERSION, int(type_code) & 0xFF, len(payload), sequence & 0xFFFFFFFF)
    return _CUE_MAGIC + header + payload


def thumb_sector(
    thumb_x: float,
    thumb_y: float,
    *,
    deadzone: float = _THUMB_DEADZONE,
) -> str | None:
    """Return left/right/up/down when the stick is inside that 45° wedge, else None.

    Gaps between wedges and the deadzone both return None. Callers that must
    tell those apart should also check the magnitude.
    """
    mag = math.hypot(float(thumb_x), float(thumb_y))
    if mag < deadzone:
        return None
    angle = math.degrees(math.atan2(float(thumb_y), float(thumb_x))) % 360.0
    for name, center in _THUMB_SECTOR_CENTER_DEG.items():
        diff = abs((angle - center + 180.0) % 360.0 - 180.0)
        if diff <= _THUMB_HALF_SECTOR_DEG:
            return name
    return None


def thumb_down_pressed(thumb_x: float, thumb_y: float, *, deadzone: float = _THUMB_DEADZONE) -> bool:
    """True when stick is in the down 45° sector with magnitude above deadzone."""
    return thumb_sector(thumb_x, thumb_y, deadzone=deadzone) == "down"


class ButtonEdgeEngine:
    """Short/long press detector matching teleop TriggerEngine BOTH/SHORT_ONLY."""

    def __init__(self, *, long_press_sec: float = 0.8) -> None:
        self._long_press_sec = max(0.1, float(long_press_sec))
        self._sources: dict[str, _SourceState] = {}
        self._last_time = 0.0

    def register(self, source: str, *, long_mode: bool) -> None:
        st = self._sources.get(source)
        if st is None:
            self._sources[source] = _SourceState(long_mode=long_mode)
        else:
            st.long_mode = st.long_mode or long_mode

    def update(self, t: float, values: dict[str, bool]) -> bool:
        # Duplicate / non-increasing device timestamps come from polling the same
        # VR snapshot twice. Skipping (instead of resetting) matches keyboard:
        # one physical hold must not re-arm short_press. Caller must not read
        # short_press/long_press when this returns False — edges stay sticky.
        if t <= self._last_time and self._last_time > 0.0:
            return False
        self._last_time = t
        for source, st in self._sources.items():
            pressed = bool(values.get(source, False))
            st.last_state = st.state
            if st.state == _PressState.RELEASED:
                if pressed:
                    st.state = _PressState.PRESSED
                    st.press_start_time = t
                    st.long_press_triggered = False
            elif st.state == _PressState.PRESSED:
                if not pressed:
                    st.state = _PressState.RELEASED
                elif t - st.press_start_time >= self._long_press_sec:
                    st.state = _PressState.LONG_PRESSED
                    st.long_press_triggered = True
            elif st.state == _PressState.LONG_PRESSED:
                if not pressed:
                    st.state = _PressState.RELEASED
        return True

    def reset(self) -> None:
        for st in self._sources.values():
            st.state = _PressState.RELEASED
            st.last_state = _PressState.RELEASED
            st.press_start_time = 0.0
            st.long_press_triggered = False
        self._last_time = 0.0

    def short_press(self, source: str) -> bool:
        st = self._sources.get(source)
        if st is None:
            return False
        if st.long_mode:
            return (
                st.last_state == _PressState.PRESSED
                and st.state == _PressState.RELEASED
                and not st.long_press_triggered
            )
        return st.last_state == _PressState.RELEASED and st.state == _PressState.PRESSED

    def long_press(self, source: str) -> bool:
        st = self._sources.get(source)
        if st is None:
            return False
        return st.last_state == _PressState.PRESSED and st.state == _PressState.LONG_PRESSED


class VrManualActionMapper:
    """Map left controller buttons/stick to (action, cue_code) events.

    Y press starts full task; the whole episode is critical. Stick left / right
    / down are success / failure / home; up toggles drag teaching. A stick direction fires once and stays
    latched until the stick returns inside the deadzone, so a sweep cannot emit
    a second ending.

    Face buttons suppress stick endings: pressing X/Y often nudges the stick, and
    keyboard ``c`` never co-fires with ``s``/``f``/``h``.
    """

    def __init__(self, *, long_press_sec: float = 0.8) -> None:
        self._engine = ButtonEdgeEngine(long_press_sec=long_press_sec)
        self._engine.register("left_y", long_mode=False)
        self._engine.register("left_x", long_mode=False)
        self._engine.register("thumb_left", long_mode=False)
        self._engine.register("thumb_right", long_mode=False)
        self._engine.register("thumb_down", long_mode=False)
        self._engine.register("thumb_up", long_mode=False)
        self._stick_latched = False
        self._drag_armed = False

    def reset(self) -> None:
        self._engine.reset()
        self._stick_latched = False
        self._drag_armed = False

    def on_buttons(
        self,
        *,
        t: float,
        left_x: int | bool,
        left_y: int | bool,
        thumb_x: float,
        thumb_y: float,
    ) -> list[tuple[str, int]]:
        mag = math.hypot(float(thumb_x), float(thumb_y))
        sector = thumb_sector(thumb_x, thumb_y)
        face_held = bool(left_x) or bool(left_y)
        if mag < _THUMB_DEADZONE:
            self._stick_latched = False
            self._drag_armed = True
        active = None
        if face_held:
            # Thumb is on X/Y: ignore stick and latch any deflection so releasing
            # the button cannot immediately fire success/failure/home.
            if sector is not None:
                self._stick_latched = True
        elif sector is not None and not self._stick_latched:
            active = sector if sector != "up" or self._drag_armed else None
            self._stick_latched = True
        advanced = self._engine.update(
            float(t),
            {
                "left_x": bool(left_x),
                "left_y": bool(left_y),
                "thumb_left": active == "left",
                "thumb_right": active == "right",
                "thumb_down": active == "down",
                "thumb_up": active == "up",
            },
        )
        if not advanced:
            return []
        out: list[tuple[str, int]] = []
        if self._engine.short_press("left_y"):
            out.append(("next", CUE_NEXT))
        if self._engine.short_press("left_x"):
            out.append(("critical", CUE_CRITICAL))
        if self._engine.short_press("thumb_left"):
            out.append(("success", CUE_SUCCESS))
        if self._engine.short_press("thumb_right"):
            out.append(("failure", CUE_FAILURE))
        if self._engine.short_press("thumb_up"):
            out.append(("drag_toggle", CUE_CRITICAL))
        if self._engine.short_press("thumb_down"):
            out.append(("home", CUE_HOME))
        return out


def gripper_opening_from_index_trigger(index_trigger: float, max_opening_m: float = 0.09) -> float:
    """Teleop grip map: released index trigger is open, fully squeezed is closed."""
    trigger = float(index_trigger)
    if trigger < 0.0:
        trigger = 0.0
    elif trigger > 1.0:
        trigger = 1.0
    return (1.0 - trigger) * float(max_opening_m)


def left_index_trigger(snap: Any) -> float | None:
    frame = getattr(snap, "frame", None)
    if frame is None:
        return None
    left = getattr(frame, "left_keys", None)
    if left is None:
        return None
    return float(getattr(left, "index_trigger", 0.0))


def process_vr_snapshot(
    snap: Any,
    *,
    mapper: VrManualActionMapper,
    last_ref: tuple[int, int] | None,
) -> tuple[list[tuple[str, int]], tuple[int, int] | None]:
    """Advance mapper only on a new receiver (epoch, sequence), like one key event."""
    frame = getattr(snap, "frame", None)
    sequence = int(getattr(snap, "sequence", 0) or 0)
    epoch = int(getattr(snap, "epoch", 0) or 0)
    if frame is None or sequence <= 0:
        return [], last_ref
    ref = (epoch, sequence)
    if last_ref is not None and ref == last_ref:
        return [], last_ref
    if last_ref is not None and epoch != last_ref[0]:
        mapper.reset()
    left = frame.left_keys
    actions = mapper.on_buttons(
        t=float(getattr(frame, "timestamp", time.time())),
        left_x=int(left.one),
        left_y=int(left.two),
        thumb_x=float(left.thumb_x),
        thumb_y=float(left.thumb_y),
    )
    return actions, ref


def apply_vr_actions(
    actions: list[tuple[str, int]],
    *,
    runtime: Any,
    home_key: Any | None,
    send_cue: Callable[[int], None] | None,
) -> None:
    for name, cue in actions:
        if name == "next":
            start = getattr(runtime, "request_full_task_start", None)
            if callable(start):
                start()
            else:
                runtime.request_next_episode()
        elif name == "critical":
            start = getattr(runtime, "request_critical_start", None)
            if callable(start):
                start()
            else:
                runtime.enter_critical_phase()
        elif name == "drag_toggle":
            runtime.toggle_drag_teach()
        elif name == "success":
            runtime.mark_manual_success()
        elif name == "failure":
            runtime.mark_manual_failure()
        elif name == "home":
            if home_key is not None and getattr(home_key, "requested", None) is not None:
                home_key.requested.set()
        if send_cue is not None:
            try:
                send_cue(int(cue))
            except Exception:
                LOG.debug("cue send failed for %s/%s", name, cue, exc_info=True)


def _find_tuozhuai() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("XR1_TUOZHUAI_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path("/opt/deeptouch"))
    here = Path(__file__).resolve()
    for parent in (here.parents[2], *here.parents):
        bundled = parent / "opt" / "deeptouch-tuozhuai"
        if bundled.is_dir():
            candidates.append(bundled)
    for candidate in candidates:
        teleop_src = candidate / "deeptouch_teleop" / "src"
        quest = teleop_src / "deeptouch_teleop" / "devices" / "vr" / "quest" / "device.py"
        # tuozhuai layout uses src/deeptouch_teleop/... OR flat src/devices/...
        flat = teleop_src / "devices" / "vr" / "quest" / "device.py"
        if quest.is_file() or flat.is_file():
            return candidate
    return None


def _ensure_teleop_on_path(tuozhuai: Path) -> Path:
    """Return teleop src dir and put it on sys.path (tuozhuai uses bare imports)."""
    teleop_src = tuozhuai / "deeptouch_teleop" / "src"
    # Prefer package layout; fall back to src root used by older trees.
    pkg = teleop_src / "deeptouch_teleop"
    if (pkg / "devices" / "vr" / "quest" / "device.py").is_file():
        # Need both package parent and inner for `devices.*` absolute imports.
        for path in (str(teleop_src), str(pkg)):
            if path not in sys.path:
                sys.path.insert(0, path)
        return pkg
    flat_device = teleop_src / "devices" / "vr" / "quest" / "device.py"
    if flat_device.is_file():
        if str(teleop_src) not in sys.path:
            sys.path.insert(0, str(teleop_src))
        return teleop_src
    raise ImportError(f"deeptouch_teleop quest device not found under {tuozhuai}")


def device_raw_snapshot(device: Any) -> Any:
    """Compatible with /opt VRDevice.raw_snapshot and older .snapshot()."""
    raw = getattr(device, "raw_snapshot", None)
    if callable(raw):
        return raw()
    snap = getattr(device, "snapshot", None)
    if callable(snap):
        return snap()
    raise AttributeError(f"{type(device).__name__} has neither raw_snapshot nor snapshot")


def device_send_bytes(device: Any, payload: bytes) -> bool:
    """Compatible with VRDevice.send_bytes or receiver.send_bytes (/opt)."""
    send = getattr(device, "send_bytes", None)
    if callable(send):
        return bool(send(payload))
    receiver = getattr(device, "_receiver", None)
    if receiver is not None:
        send_r = getattr(receiver, "send_bytes", None)
        if callable(send_r):
            return bool(send_r(payload))
    return False


class _QuestCueSender:
    def __init__(self, device: Any) -> None:
        self._device = device
        self._sequence = 0

    def __call__(self, type_code: int) -> None:
        self._sequence += 1
        frame = encode_quest_cue(type_code, self._sequence)
        try:
            ok = device_send_bytes(self._device, frame)
            if not ok:
                LOG.debug("headset cue %s dropped (device offline)", type_code)
        except Exception as exc:
            LOG.debug("headset cue %s send failed: %s", type_code, exc)


def start_vr_manual_controls(
    runtime: Any,
    home_key_provider: Callable[[], Any | None],
    *,
    enabled: bool = True,
    host: str = "127.0.0.1",
    port: int = 40051,
    long_press_sec: float = 0.8,
    poll_hz: float = 60.0,
) -> threading.Thread | None:
    """Start background VR reader. Returns None if disabled or soft-failed at import/start."""
    if not enabled:
        return None

    try:
        tuozhuai = _find_tuozhuai()
        if tuozhuai is None:
            LOG.warning("VR controls: tuozhuai not found; keyboard-only")
            return None
        _ensure_teleop_on_path(tuozhuai)
        try:
            from deeptouch_teleop.devices.vr.quest.device import VRDevice  # type: ignore
        except ImportError:
            from devices.vr.quest.device import VRDevice  # type: ignore
    except Exception as exc:
        LOG.warning("VR controls: cannot import VRDevice (%s); keyboard-only", exc)
        return None

    device = VRDevice(host=host, port=port, mode="local")
    mapper = VrManualActionMapper(long_press_sec=long_press_sec)

    def loop() -> None:
        try:
            if not device.start():
                LOG.warning("VR controls: VRDevice.start() failed (ADB/Quest); keyboard-only")
                return
        except Exception as exc:
            LOG.warning("VR controls: start raised (%s); keyboard-only", exc)
            return
        LOG.info(
            "VR controls ready (USB local %s:%d): "
            "index trigger=left gripper, Y=full task (whole episode critical), "
            "stick-left=s, stick-right=f, stick-down=h, stick-up=drag toggle",
            host,
            port,
        )
        send_cue = _QuestCueSender(device)
        interval = 1.0 / max(1.0, float(poll_hz))
        last_ref: tuple[int, int] | None = None
        try:
            while not bool(getattr(runtime, "shutdown_requested", lambda: False)()):
                snap = device_raw_snapshot(device)
                trigger = left_index_trigger(snap)
                if trigger is not None:
                    publish = getattr(runtime, "set_left_index_trigger", None)
                    if callable(publish):
                        publish(trigger)
                actions, last_ref = process_vr_snapshot(snap, mapper=mapper, last_ref=last_ref)
                if actions:
                    apply_vr_actions(
                        actions,
                        runtime=runtime,
                        home_key=home_key_provider(),
                        send_cue=send_cue,
                    )
                time.sleep(interval)
        finally:
            try:
                device.stop()
            except Exception:
                LOG.debug("VRDevice.stop failed", exc_info=True)

    thread = threading.Thread(target=loop, name="deeptouch-online-rl-vr", daemon=True)
    thread.start()
    return thread
