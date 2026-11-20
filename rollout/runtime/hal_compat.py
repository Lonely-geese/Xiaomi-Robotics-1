"""Experimental-client-only HAL stop negotiation and measured-stop checks."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


def select_stop_backend(capabilities):
    """Use advertised operations, never an optimistic motion-command probe."""
    ops = {op["name"]: op for op in capabilities.get("operations", [])}
    native = ops.get("stop_online_tracking", {})
    params = {p["name"]: p for p in native.get("params", [])}
    if native.get("blocking") is True and "unit" in params:
        choices = params["unit"].get("enum_options", [])
        if not choices or "leftarm" in choices:
            return "native_stop_operation"
    mode = ops.get("set_motion_mode", {})
    params = {p["name"]: p for p in mode.get("params", [])}
    # The fallback sends the public SET_MOTION_MODE RPC with units=['leftarm'].
    # Require explicit per-unit support; never fall back to a both-arm switch.
    if ("units" not in params or params["units"].get("type") != "string" or
            "trajectory" not in params.get("mode", {}).get("enum_options", []) or
            "cart" not in params.get("command_space", {}).get("enum_options", [])):
        raise ValueError("HAL has no advertised left-unit stop/mode-switch capability")
    states = {s["state_key"]: s for s in capabilities.get("state_channels", [])}
    for key, unit in (("leftarm_state_joint_pos", "degree"),
                      ("leftarm_state_cart_pos", "m+quaternion")):
        s = states.get(key, {})
        if s.get("state_shape") != [7] or s.get("unit") != unit:
            raise ValueError(f"HAL stop feedback needs {key} shape [7] and unit {unit}")
    return "mode_switch_feedback"


@dataclass(frozen=True)
class StopLimits:
    timeout: float = 5.
    stable_s: float = .2
    feedback_age: float = .15
    linear_speed: float = .005
    angular_speed: float = math.radians(1)
    joint_speed: float = 3.
    position_drift: float = .04
    rotation_drift: float = math.radians(15)


class ArmStopped:
    """Require fresh post-request feedback, a 100ms window and sustained rest.

    This confirms measured rest, not the controller's internal stop-complete
    flag. HAL cache timestamps are not device hardware timestamps.
    """
    def __init__(self, anchor, command_time, limits):
        self.anchor = np.asarray(anchor, float).copy()
        self.command_time, self.limits = command_time, limits
        self.rows = deque()
        self.sequence = None
        self.stable_since = None

    @staticmethod
    def angle(a, b):
        return 2*math.acos(float(np.clip(abs(np.dot(a, b)), 0, 1)))

    def update(self, pose, joints, stamp, sequence, now):
        lim = self.limits
        pose, joints = np.asarray(pose, float).copy(), np.asarray(joints, float).copy()
        if (pose.shape != (7,) or joints.shape != (7,) or
                not np.isfinite(pose).all() or not np.isfinite(joints).all() or
                abs(np.linalg.norm(pose[3:])-1) > .001):
            raise ValueError("invalid arm feedback during HAL stop")
        if not math.isfinite(stamp) or not -.01 <= now-stamp <= lim.feedback_age:
            raise ValueError("stale arm feedback during HAL stop")
        if (np.linalg.norm(pose[:3]-self.anchor[:3]) > lim.position_drift or
                self.angle(pose[3:], self.anchor[3:]) > lim.rotation_drift):
            raise ValueError("arm drift exceeded following protection during HAL stop")
        if stamp <= self.command_time or sequence == self.sequence:
            return {"confirmed": False, "fresh": False}
        if self.rows and stamp <= self.rows[-1][0]:
            raise ValueError("HAL stop feedback timestamp did not advance")
        if self.rows and stamp-self.rows[-1][0] > lim.feedback_age:
            self.rows.clear()
            self.stable_since = None
        self.sequence = sequence
        self.rows.append((stamp, pose, joints))
        while len(self.rows) > 2 and stamp-self.rows[1][0] >= .1:
            self.rows.popleft()
        dt = stamp-self.rows[0][0]
        speeds = {}
        if dt >= .1-1e-9:
            rows = list(self.rows)
            speeds = {
                "linear_speed_m_s": float(sum(np.linalg.norm(b[1][:3]-a[1][:3])
                    for a, b in zip(rows, rows[1:]))/dt),
                "angular_speed_rad_s": sum(self.angle(a[1][3:], b[1][3:])
                    for a, b in zip(rows, rows[1:]))/dt,
                "joint_speed_deg_s": float(np.max(np.abs(np.diff(
                    [r[2] for r in rows], axis=0)).sum(axis=0))/dt),
            }
        within = (bool(speeds) and speeds["linear_speed_m_s"] <= lim.linear_speed and
                  speeds["angular_speed_rad_s"] <= lim.angular_speed and
                  speeds["joint_speed_deg_s"] <= lim.joint_speed)
        if not within:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = stamp
        stable_s = 0. if self.stable_since is None else stamp-self.stable_since
        return {"fresh": True, "confirmed": within and stable_s >= lim.stable_s,
                "stable_s": stable_s, "feedback_sequence": sequence,
                "feedback_stamp": stamp, **speeds}
