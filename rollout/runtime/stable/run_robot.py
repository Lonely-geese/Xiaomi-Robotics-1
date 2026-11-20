#!/usr/bin/env python3
"""XR1 smooth Cartesian policy with Xiaomi action-prefix asynchronous inference."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent


def _find_bundled_tuozhuai():
    for parent in (BUNDLE, *BUNDLE.parents):
        candidate = parent / "opt" / "deeptouch-tuozhuai"
        if (candidate / "deeptouch_robot_hal/src/robot_hal").is_dir():
            return candidate
    return None


def _select_tuozhuai():
    candidates = []
    configured = os.environ.get("XR1_TUOZHUAI_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path("/opt/deeptouch"))
    bundled = _find_bundled_tuozhuai()
    if bundled is not None:
        candidates.append(bundled)
    for candidate in candidates:
        if (candidate / "deeptouch_robot_hal/src/robot_hal").is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"cannot find a robot_hal runtime; searched: {searched}")


TUOZHUAI = _select_tuozhuai()
HAL_ROOT = TUOZHUAI / "deeptouch_robot_hal"
for path in (HAL_ROOT / "src", ROOT):
    sys.path.insert(0, str(path))

from wire import WIRE_FORMAT, connect as connect_policy, recv_arrays, send_arrays
from tianji_image_decoder import decode_observation_frames

# The model environment deliberately does not carry the robot HAL stack.  Keep
# pure conversion/scheduling code importable for offline validation while the
# robot entry point still fails immediately if its own environment is incomplete.
try:
    from robot_hal.client import RobotHalClient
    from robot_hal.protocol.types import CommandSpace, ControlMode, MotionMode
    _ROBOT_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    RobotHalClient = CommandSpace = ControlMode = MotionMode = None
    _ROBOT_IMPORT_ERROR = error

LOG = logging.getLogger("xr1.cartesian.robot")
CAMERAS = {"cam_high": "head_rgb", "cam_left_wrist": "left_wrist_rgb",
           "cam_right_wrist": "right_wrist_rgb"}
STATE_KEYS = {
    "left_arm": "leftarm_state_joint_pos", "right_arm": "rightarm_state_joint_pos",
    "left_gripper": "leftarm_gripper_state_pos", "right_gripper": "rightarm_gripper_state_pos",
    "left_cart": "leftarm_state_cart_pos", "right_cart": "rightarm_state_cart_pos",
}
FEEDBACK_KEYS = ["leftarm_state_cart_pos", "leftarm_state_joint_pos",
                 "leftarm_gripper_state_pos", "rightarm_gripper_state_pos"]
_reset_candidates = []
if os.environ.get("XR1_RESET_CONFIG"):
    _reset_candidates.append(Path(os.environ["XR1_RESET_CONFIG"]).expanduser())
_reset_candidates.extend([
    TUOZHUAI / "deeptouch_teleop/config/vr_teleop.yaml",
    (_find_bundled_tuozhuai() or TUOZHUAI) / "deeptouch_teleop/config/vr_teleop.yaml",
])
RESET_CONFIG = next((path for path in _reset_candidates if path.is_file()), None)
INITIAL_GRIPPER_M = .09
GRIPPER_COMMAND_TOLERANCE_M = 1e-6


def _expected_policy_identity():
    configured = os.environ.get("XR1_MODEL_MANIFEST")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(ROOT / "active_model/model_manifest.json")
    candidates.extend(parent / "manifest.json" for parent in ROOT.parents)
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        raise RuntimeError("cannot find the active Xiaomi model manifest")
    manifest = json.loads(manifest_path.read_text())
    required = ("checkpoint_step", "checkpoint_sha256")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise RuntimeError(f"active Xiaomi model manifest is missing {missing}")
    return {
        "checkpoint_step": manifest["checkpoint_step"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_epoch": manifest.get("checkpoint_epoch"),
        "completed_epochs": manifest.get("completed_epochs"),
        "model_profile": manifest.get("profile", manifest.get("bundle")),
    }


class HardwareError(RuntimeError):
    pass


def normalize_quaternion(quaternion, name="quaternion"):
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise HardwareError(f"{name} must be finite XYZW")
    norm = np.linalg.norm(value)
    if abs(norm - 1) > 1e-3:
        raise HardwareError(f"{name} is not unit length: {norm}")
    return value / norm


def quaternion_angle(left, right):
    return 2 * math.acos(float(np.clip(abs(np.dot(normalize_quaternion(left),
                                                  normalize_quaternion(right))), 0, 1)))


def _quintic_hermite(position0, position1, velocity0, velocity1,
                      acceleration0, acceleration1, duration, fraction):
    """Evaluate one quintic segment with shared position/velocity/acceleration."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("quintic duration must be finite and positive")
    fraction = float(fraction)
    if not 0 <= fraction <= 1:
        raise ValueError("quintic fraction must be in 0..1")
    p0 = np.asarray(position0, dtype=np.float64)
    p1 = np.asarray(position1, dtype=np.float64)
    v0 = np.asarray(velocity0, dtype=np.float64) * duration
    v1 = np.asarray(velocity1, dtype=np.float64) * duration
    a0 = np.asarray(acceleration0, dtype=np.float64) * duration**2
    a1 = np.asarray(acceleration1, dtype=np.float64) * duration**2
    u = fraction
    u2, u3, u4, u5 = u*u, u*u*u, u*u*u*u, u*u*u*u*u
    h00 = 1 - 10*u3 + 15*u4 - 6*u5
    h01 = 10*u3 - 15*u4 + 6*u5
    h10 = u - 6*u3 + 8*u4 - 3*u5
    h11 = -4*u3 + 7*u4 - 3*u5
    h20 = .5*(u2 - 3*u3 + 3*u4 - u5)
    h21 = .5*(u3 - 2*u4 + u5)
    return h00*p0 + h01*p1 + h10*v0 + h11*v1 + h20*a0 + h21*a1


def _c2_knot_derivatives(values, interval_s):
    """Estimate derivatives once per knot; only complete-path ends are stationary."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("C2 spline values must be a finite 2-D sequence")
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("C2 spline interval must be finite and positive")
    velocities = np.zeros_like(values)
    accelerations = np.zeros_like(values)
    if len(values) > 2:
        velocities[1:-1] = (values[2:] - values[:-2]) / (2*interval_s)
        accelerations[1:-1] = (
            values[2:] - 2*values[1:-1] + values[:-2]) / interval_s**2
    return velocities, accelerations


def _c2_quintic_samples(values, interval_s, steps):
    values = np.asarray(values, dtype=np.float64)
    velocities, accelerations = _c2_knot_derivatives(values, interval_s)
    samples = []
    for index in range(len(values)-1):
        for step in range(1, steps+1):
            samples.append(_quintic_hermite(
                values[index], values[index+1],
                velocities[index], velocities[index+1],
                accelerations[index], accelerations[index+1],
                interval_s, step/steps))
    return np.asarray(samples, dtype=np.float64), velocities, accelerations


def interpolate_cartesian_targets(start_pose, poses, policy_hz, servo_hz):
    """Resample policy keyframes without changing their timing or endpoints.

    The measured/last commanded pose is the t=0 anchor. Every policy pose remains
    an exact keyframe at its original time. Interior knots share velocity and
    acceleration; only the complete chunk boundaries are stationary. Quaternion
    signs are aligned before interpolation to avoid taking the antipodal branch.
    """
    start_pose = np.asarray(start_pose, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    if (start_pose.shape != (7,) or poses.ndim != 2 or poses.shape[1] != 7 or
            not len(poses) or not np.isfinite(start_pose).all() or
            not np.isfinite(poses).all()):
        raise HardwareError("invalid Cartesian policy targets for interpolation")
    ratio = float(servo_hz) / float(policy_hz)
    steps = int(round(ratio))
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=0, abs_tol=1e-9):
        raise HardwareError("--servo-hz must be an integer multiple of --control-hz")

    key_poses = np.vstack([start_pose, poses]).astype(np.float64, copy=True)
    key_poses[:, 3:] = np.stack([
        normalize_quaternion(quaternion) for quaternion in key_poses[:, 3:]
    ])
    for index in range(1, len(key_poses)):
        if np.dot(key_poses[index-1, 3:], key_poses[index, 3:]) < 0:
            key_poses[index, 3:] *= -1

    interval_s = 1/float(policy_hz)
    dense_xyz, xyz_velocity, xyz_acceleration = _c2_quintic_samples(
        key_poses[:, :3], interval_s, steps)
    dense_quaternion_raw, quaternion_velocity, quaternion_acceleration = (
        _c2_quintic_samples(key_poses[:, 3:], interval_s, steps))
    dense_quaternion = np.stack([
        normalize_quaternion(quaternion, "interpolated quaternion")
        for quaternion in dense_quaternion_raw
    ])
    dense_pose = np.column_stack([dense_xyz, dense_quaternion])
    policy_indices = np.repeat(np.arange(len(poses), dtype=np.int64), steps)
    interpolation_steps = np.tile(
        np.arange(1, steps+1, dtype=np.int64), len(poses))
    return {
        "pose": dense_pose,
        "policy_index": policy_indices,
        "interpolation_step": interpolation_steps,
        "interpolation_steps": steps,
        "spline": "c2_quintic_stationary_chunk_boundary",
        "boundary_velocity_zero": True,
        "boundary_acceleration_zero": True,
        "xyz_knot_velocity": xyz_velocity,
        "xyz_knot_acceleration": xyz_acceleration,
        "quaternion_knot_velocity": quaternion_velocity,
        "quaternion_knot_acceleration": quaternion_acceleration,
    }


def quaternion_matrix(quaternion):
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def quaternion_multiply(left, right):
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2,
                     w1*w2-x1*x2-y1*y2-z1*z2])


def relative_rotvec(observed, target):
    observed = normalize_quaternion(observed)
    target = normalize_quaternion(target)
    relative = quaternion_multiply(np.r_[-observed[:3], observed[3]], target)
    if relative[3] < 0:
        relative = -relative
    vector_norm = np.linalg.norm(relative[:3])
    if vector_norm < 1e-10:
        return np.zeros(3)
    angle = 2 * math.atan2(vector_norm, float(relative[3]))
    return relative[:3] * (angle / vector_norm)


def image_chw(frame):
    image = np.asarray(frame)
    if image.ndim != 3:
        raise HardwareError(f"camera must be 3-D, got {image.shape}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3]
    elif image.shape[0] in (3, 4):
        image = image[:3].transpose(1, 2, 0)
    else:
        raise HardwareError(f"camera must be RGB HWC/CHW, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if float(image.mean()) < 5:
        raise HardwareError("camera image is nearly black")
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def build_raw(state, task):
    raw_state = {}
    for name, key in STATE_KEYS.items():
        shape = (7,) if name.endswith("arm") or name.endswith("cart") else (1,)
        value = np.asarray(state.get(key), dtype=np.float32).reshape(-1)
        if value.shape != shape or not np.isfinite(value).all():
            raise HardwareError(f"invalid/missing {key}: {value.shape}")
        raw_state[name] = value
    for side in ("left", "right"):
        normalize_quaternion(raw_state[f"{side}_cart"][3:], f"{side} pose quaternion")
    images = {model_key: image_chw(state[hal_key]) for hal_key, model_key in CAMERAS.items()}
    return {"images": images, "state": raw_state, "task": task, "frequency": 30}


def decode_feedback(state):
    pose = np.asarray(state.get(FEEDBACK_KEYS[0]), dtype=np.float64).reshape(-1)
    joints = np.asarray(state.get(FEEDBACK_KEYS[1]), dtype=np.float64).reshape(-1)
    left_grip = np.asarray(state.get(FEEDBACK_KEYS[2]), dtype=np.float64).reshape(-1)
    right_grip = np.asarray(state.get(FEEDBACK_KEYS[3]), dtype=np.float64).reshape(-1)
    values = (pose, joints, left_grip, right_grip)
    if ([value.shape for value in values] != [(7,), (7,), (1,), (1,)] or
            not all(np.isfinite(value).all() for value in values)):
        raise HardwareError("invalid Cartesian/joint/gripper feedback")
    normalize_quaternion(pose[3:])
    return pose, joints, float(left_grip[0]), float(right_grip[0])


class ObservationSource:
    """Continuously consume aligned HAL observations so control never waits for images."""
    def __init__(self, host, port, max_age):
        if RobotHalClient is None:
            raise RuntimeError("robot client environment is incomplete") from _ROBOT_IMPORT_ERROR
        self.client = RobotHalClient(host=host, port=port, transport="shm")
        self.max_age = max_age
        self.stream = None
        self.latest_value = None
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="xr1-observation", daemon=True)

    def start(self):
        self.client.connect()
        self.stream = self.client.open_observation_stream("latest")
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                result = self.stream.read_next(.2)
                status = result.get("status")
                if status in ("timeout", "not_yet_available", "subscriber_gap"):
                    continue
                if status in ("system_fault", "stream_epoch_changed"):
                    raise HardwareError(result.get("fault") or status)
                observation = result.get("observation")
                if observation is None:
                    continue
                receive = (((observation.get("metadata") or {}).get("reference_timestamps") or {})
                           .get("receive") or {})
                if receive.get("clock_id") != "hal_monotonic" or "time_ns" not in receive:
                    raise HardwareError("observation lacks hal_monotonic timestamp")
                stamp = int(receive["time_ns"]) / 1e9
                decoded = dict(observation["state"])
                decoded.update(decode_observation_frames(observation["camera_frames"]))
                with self.condition:
                    self.latest_value = (decoded, stamp, observation.get("sequence"))
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def latest(self, timeout=5, min_stamp=None):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise HardwareError("observation reader failed") from self.error
                if self.latest_value is not None:
                    state, stamp, sequence = self.latest_value
                    age = time.monotonic() - stamp
                    if (-.01 <= age <= self.max_age and
                            (min_stamp is None or stamp >= min_stamp)):
                        return dict(state), stamp, sequence
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HardwareError("no fresh aligned observation")
                self.condition.wait(min(.1, remaining))

    def close(self):
        self.stop.set()
        if self.stream is not None:
            self.stream.close()
        self.thread.join(2)
        self.client.close()


class CartesianHAL:
    def __init__(self, host, port, feedback_hz=60, max_feedback_age=.15):
        if RobotHalClient is None:
            raise RuntimeError("robot client environment is incomplete") from _ROBOT_IMPORT_ERROR
        # Commands and feedback use independent RPC sessions. A slow GET_STATE
        # can no longer head-of-line block a Cartesian command on the same socket.
        self.client = RobotHalClient(host=host, port=port, transport="tcp")
        self.feedback_client = RobotHalClient(host=host, port=port, transport="tcp")
        self.feedback_hz = float(feedback_hz)
        self.max_feedback_age = float(max_feedback_age)
        self._feedback_condition = threading.Condition()
        self._feedback_stop = threading.Event()
        self._feedback_value = None
        self._feedback_error = None
        self._feedback_sequence = 0
        self._last_feedback_recovery = None
        self._feedback_thread = threading.Thread(
            target=self._feedback_loop, name="xr1-hal-feedback", daemon=True)
        self.have_control = False
        self.tracking = False
        self.right_gripper_hold = .09
        self.cap_linear = self.cap_angular = None

    def connect(self):
        self.client.connect()
        capabilities = self.client.get_capabilities()
        if capabilities["pose_format"] != {"position_unit": "m", "quat_order": "xyzw"}:
            raise HardwareError(f"unsupported HAL pose format: {capabilities['pose_format']}")
        channels = {channel["cmd_key"]: channel for channel in capabilities["cmd_channels"]}
        cart = channels.get("leftarm_cmd_cart_pos")
        if (not cart or cart["cmd_shape"] != [7] or cart["cmd_space"] != "cartesian" or
                int(cart["required_command_space"]) != int(CommandSpace.CART)):
            version = self.client.server_version or "unknown"
            raise HardwareError(
                f"HAL {version} has no compatible left Cartesian command channel; "
                "Apex arm may be offline or the HAL is not the current Cartesian build"
            )
        self.cap_linear = float(cart["max_linear_speed_m_s"])
        self.cap_angular = float(cart["max_angular_speed_rad_s"])
        if (not math.isfinite(self.cap_linear) or self.cap_linear <= 0 or
                not math.isfinite(self.cap_angular) or self.cap_angular <= 0):
            raise HardwareError(
                f"invalid left Cartesian capabilities: "
                f"linear={self.cap_linear} angular={self.cap_angular}"
            )
        state_keys = {channel["state_key"] for channel in capabilities["state_channels"]}
        required_state = {
            "leftarm_state_cart_pos", "leftarm_state_joint_pos",
            "rightarm_state_joint_pos", "leftarm_gripper_state_pos",
            "rightarm_gripper_state_pos",
        }
        missing_state = sorted(required_state - state_keys)
        if missing_state:
            raise HardwareError(
                f"HAL {self.client.server_version or 'unknown'} is missing required "
                f"XR1 state channels: {missing_state}; Apex arm/gripper state is offline"
            )
        required_commands = {"leftarm_gripper_cmd_pos", "rightarm_gripper_cmd_pos"}
        missing_commands = sorted(required_commands - set(channels))
        if missing_commands:
            raise HardwareError(
                f"HAL is missing required XR1 gripper command channels: {missing_commands}"
            )
        self.feedback_client.connect()
        self._feedback_thread.start()
        self.feedback_with_metadata(timeout=2)
        LOG.info("connected to HAL server=%s runtime=%s", self.client.server_version, TUOZHUAI)

    def acquire(self):
        self.client.acquire_control(ControlMode.POLICY)
        self.have_control = True

    def _feedback_loop(self):
        period = 1 / self.feedback_hz
        while not self._feedback_stop.is_set():
            try:
                rpc_started = time.monotonic()
                result = decode_feedback(self.feedback_client.get(FEEDBACK_KEYS))
                stamp = time.monotonic()
                rpc_s = stamp-rpc_started
            except BaseException as error:
                if self._feedback_stop.is_set():
                    return
                with self._feedback_condition:
                    self._feedback_error = error
                    self._feedback_condition.notify_all()
                return
            with self._feedback_condition:
                self._feedback_sequence += 1
                self._feedback_value = (*result, stamp, self._feedback_sequence, rpc_s)
                self._feedback_condition.notify_all()
            self._feedback_stop.wait(period)

    def feedback_with_metadata(self, timeout=1):
        deadline = time.monotonic() + timeout
        recovery = None
        with self._feedback_condition:
            while self._feedback_value is None and self._feedback_error is None:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise HardwareError("no HAL feedback sample")
                self._feedback_condition.wait(remaining)
            if self._feedback_error is not None:
                raise HardwareError("HAL feedback reader failed") from self._feedback_error
            pose, joints, left_grip, right_grip, stamp, sequence, rpc_s = self._feedback_value
            age = time.monotonic()-stamp
            if age > self.max_feedback_age:
                # One scheduler hiccup must not stop an otherwise healthy run.
                # Wait for exactly one independent feedback-poll opportunity;
                # continued staleness still fails closed before stale state is
                # used by the control loop.
                stale_age = age
                stale_sequence = sequence
                detected_at = time.monotonic()
                grace_s = min(1/self.feedback_hz, max(0.0, deadline-detected_at))
                if grace_s > 0:
                    self._feedback_condition.wait_for(
                        lambda: (self._feedback_error is not None or
                                 (self._feedback_value is not None and
                                  self._feedback_value[5] != stale_sequence)),
                        timeout=grace_s,
                    )
                if self._feedback_error is not None:
                    raise HardwareError("HAL feedback reader failed") from self._feedback_error
                pose, joints, left_grip, right_grip, stamp, sequence, rpc_s = self._feedback_value
                age = time.monotonic()-stamp
                if sequence == stale_sequence or age > self.max_feedback_age:
                    raise HardwareError(
                        f"HAL feedback stale by {max(stale_age, age):.3f}s "
                        f"(limit {self.max_feedback_age:.3f}s) after one "
                        f"{grace_s*1000:.1f}ms recovery window"
                    )
                recovery = {
                    "stale_sequence": int(stale_sequence),
                    "recovered_sequence": int(sequence),
                    "detected_age_s": float(stale_age),
                    "recovered_age_s": float(age),
                    "recovery_wait_s": float(time.monotonic()-detected_at),
                    "recovery_window_s": float(grace_s),
                }
                self._last_feedback_recovery = recovery
        if recovery is not None:
            LOG.warning(
                "HAL feedback recovered after transient staleness: %.3fs -> %.3fs",
                recovery["detected_age_s"], recovery["recovered_age_s"])
        self.right_gripper_hold = right_grip
        return (pose.copy(), joints.copy(), left_grip, right_grip, stamp, sequence, rpc_s)

    def consume_feedback_recovery(self):
        with self._feedback_condition:
            recovery, self._last_feedback_recovery = self._last_feedback_recovery, None
        return recovery

    def feedback(self):
        return self.feedback_with_metadata()[:4]

    def diagnostics(self):
        # HAL reports an aggregate configured mode, not per-arm controller
        # status, faults, or the controller's internal tracking target.
        result = {"source": "HAL aggregate configuration",
                  "controller_internal_state_available": False,
                  "client_tracking_requested": self.tracking,
                  "client_control_acquired": self.have_control}
        try:
            mode = self.client.get_motion_mode()
            result.update(mode=mode.mode.name, command_space=mode.command_space.name)
        except Exception as error:
            result["query_error"] = f"{type(error).__name__}: {error}"
        return result

    def reset_for_inference(self, velocity, timeout, joint_tolerance_deg,
                            gripper_tolerance_m):
        """Return both arms and both grippers to the configured task start."""
        import yaml
        if RESET_CONFIG is None:
            raise HardwareError(
                "no reset configuration found; set XR1_RESET_CONFIG to a YAML "
                "file containing reset.left_init_joint_pos/right_init_joint_pos"
            )
        config = yaml.safe_load(RESET_CONFIG.read_text())["reset"]
        left, right = config.get("left_init_joint_pos"), config.get("right_init_joint_pos")
        if not (isinstance(left, list) and len(left) == 7 and isinstance(right, list) and len(right) == 7):
            raise HardwareError(f"invalid reset joints in {RESET_CONFIG}")
        LOG.info("resetting both arms to configured initial joints")
        self.client.move_to_pose(joint_positions={"leftarm": left, "rightarm": right},
                                 vel=velocity, block=True)
        LOG.info("opening both grippers to %.3f m", INITIAL_GRIPPER_M)
        self.set_grippers(INITIAL_GRIPPER_M, INITIAL_GRIPPER_M)

        keys = ["leftarm_state_joint_pos", "rightarm_state_joint_pos",
                "leftarm_gripper_state_pos", "rightarm_gripper_state_pos"]
        deadline = time.monotonic() + timeout
        stable_since = None
        last_errors = None
        while time.monotonic() < deadline:
            state = self.client.get(keys)
            values = [np.asarray(state.get(key), dtype=np.float64).reshape(-1) for key in keys]
            if ([value.shape for value in values] == [(7,), (7,), (1,), (1,)] and
                    all(np.isfinite(value).all() for value in values)):
                errors = (float(np.max(np.abs(values[0]-left))),
                          float(np.max(np.abs(values[1]-right))),
                          abs(float(values[2][0])-INITIAL_GRIPPER_M),
                          abs(float(values[3][0])-INITIAL_GRIPPER_M))
                last_errors = errors
                reached = (max(errors[:2]) <= joint_tolerance_deg and
                           max(errors[2:]) <= gripper_tolerance_m)
                if reached:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic()-stable_since >= .2:
                        LOG.info("initial pose verified: joints<=%.3fdeg grippers<=%.4fm",
                                 max(errors[:2]), max(errors[2:]))
                        return
                else:
                    stable_since = None
            time.sleep(.03)
        raise HardwareError(f"initial pose did not converge; errors={last_errors}")

    def set_grippers(self, left, right=None):
        left, right = self._validate_grippers(left, right)
        self.client.send_cmd({"leftarm_gripper_cmd_pos": np.array([left], np.float32),
                              "rightarm_gripper_cmd_pos": np.array([right], np.float32)})
        self.right_gripper_hold = right
        return left, right

    def _validate_grippers(self, left, right=None):
        right = self.right_gripper_hold if right is None else right
        left, right = float(left), float(right)
        if (not math.isfinite(left) or not math.isfinite(right) or
                left < -GRIPPER_COMMAND_TOLERANCE_M or
                right < -GRIPPER_COMMAND_TOLERANCE_M or
                left > INITIAL_GRIPPER_M+GRIPPER_COMMAND_TOLERANCE_M or
                right > INITIAL_GRIPPER_M+GRIPPER_COMMAND_TOLERANCE_M):
            raise HardwareError("gripper command outside 0..0.09 m")
        # np.float32(0.09) is 0.090000003576... when converted back to a
        # Python float.  Accept only this numerical-scale overshoot and send a
        # value inside the physical HAL range.
        left = min(INITIAL_GRIPPER_M, max(0.0, left))
        right = min(INITIAL_GRIPPER_M, max(0.0, right))
        return left, right

    def start_tracking(self):
        current, *_ = self.feedback()
        self.client.set_motion_mode(MotionMode.ONLINE_TRACKING, CommandSpace.CART,
                                    units=["leftarm"])
        self.tracking = True
        # Cartesian mode is not seeded by HAL; seed it explicitly with measured
        # World pose before any policy target.
        self.send_pose(current)

    def stop_tracking(self):
        """Block until Apex has completed the left-arm online-tracking soft stop."""
        if not self.tracking:
            return False
        result = self.client.execute_op(
            "stop_online_tracking", {"unit": "leftarm"},
            wait=True, timeout_ms=10000)
        if not result.get("success"):
            raise HardwareError(
                "HAL failed to stop Cartesian online tracking: "
                f"{result.get('error_message') or 'unknown error'}")
        payload = result.get("result")
        if (not isinstance(payload, dict) or payload.get("mode") != "TRAJECTORY" or
                payload.get("soft_stopping") is not False):
            raise HardwareError(
                f"HAL returned invalid online-tracking stop result: {payload!r}")
        self.tracking = False
        return True

    def send_pose(self, pose):
        pose, _ = self.send_command(pose)
        return pose

    def send_command(self, pose, left_gripper=None):
        """Enqueue one atomic Cartesian frame, optionally with both grippers."""
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise HardwareError("Cartesian command must be finite [xyz, XYZW]")
        pose[3:] = normalize_quaternion(pose[3:])
        command = {"leftarm_cmd_cart_pos": pose}
        grippers = None
        if left_gripper is not None:
            grippers = self._validate_grippers(left_gripper)
            command.update({
                "leftarm_gripper_cmd_pos": np.array([grippers[0]], np.float32),
                "rightarm_gripper_cmd_pos": np.array([grippers[1]], np.float32),
            })
        self.client.send_cmd(command, active_space=CommandSpace.CART)
        if grippers is not None:
            self.right_gripper_hold = grippers[1]
        return pose.copy(), grippers

    def close(self):
        if self.tracking:
            try:
                self.client.set_motion_mode(MotionMode.TRAJECTORY, CommandSpace.CART,
                                            units=["leftarm"])
            except Exception:
                LOG.exception("failed to soft-stop Cartesian tracking")
            self.tracking = False
        if self.have_control:
            try:
                self.client.release_control()
            except Exception:
                LOG.exception("failed to release HAL control")
            self.have_control = False
        self._feedback_stop.set()
        with self._feedback_condition:
            self._feedback_condition.notify_all()
        try:
            self.feedback_client.close()
        except Exception:
            LOG.exception("failed to close HAL feedback connection")
        if self._feedback_thread.is_alive():
            self._feedback_thread.join(2)
        self.client.close()


class PolicyResponseError(RuntimeError):
    """A structured rejection returned by the policy adapter."""
    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.error_code = self.metadata.get("error_code")
        self.retryable = self.metadata.get("retryable") is True
        super().__init__(self.metadata.get("error") or
                         f"invalid policy response: {self.metadata}")


class PolicyClient:
    def __init__(self, host, port, timeout=30):
        self.socket = connect_policy(host, port, timeout)
        _, self.metadata = recv_arrays(self.socket)
        expected = {"status": "ready", "adapter": "xr1_official_prefix_cartesian_v1",
                    "action_horizon": 30, "frequency_hz": 30,
                    "active_arm": "left", "action_space": "world_cartesian_xyz_xyzw",
                    "async_prefix": True, "max_prefix_length": 6,
                    "wire_format": WIRE_FORMAT,
                    "trajectory_rejection_protocol": "structured_retry_v1"}
        expected.update(_expected_policy_identity())
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise RuntimeError(f"policy handshake mismatch: {key}={self.metadata.get(key)!r}")

    def request(self, raw, serial, action_prefix=None):
        arrays = {f"image__{key}": value for key, value in raw["images"].items()}
        arrays.update({f"state__{key}": value for key, value in raw["state"].items()})
        if action_prefix is not None:
            arrays["action_prefix"] = action_prefix
        send_started = time.monotonic()
        request_bytes = send_arrays(
            self.socket, arrays,
            {"task": raw["task"], "frequency": 30, "serial": serial})
        send_finished = time.monotonic()
        response, metadata = recv_arrays(self.socket)
        received = time.monotonic()
        if metadata.get("status") != "ok" or metadata.get("serial") != serial:
            raise PolicyResponseError(metadata)
        required = {"left_pose": (30,7), "left_gripper": (30,), "observation_pose": (7,),
                    "observation_gripper": (1,), "raw_action": (30,60)}
        for key, shape in required.items():
            value = np.asarray(response.get(key))
            if value.shape != shape or not np.isfinite(value).all():
                raise RuntimeError(f"invalid policy output {key}: {value.shape}")
        metadata = dict(metadata)
        metadata.update({"request_bytes": request_bytes,
                         "client_send_s": send_finished-send_started,
                         "client_wait_and_receive_s": received-send_finished})
        return response, metadata

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass
        self.socket.close()


@dataclass
class Request:
    serial: int
    raw: dict
    prefix: np.ndarray | None
    committed_pose: np.ndarray | None
    committed_gripper: np.ndarray | None


class PolicyInferenceError(RuntimeError):
    """Bind a background inference failure to the request that produced it."""
    def __init__(self, request, error):
        self.request = request
        self.original_error = error
        self.metadata = dict(getattr(error, "metadata", {}))
        self.error_code = getattr(error, "error_code", None)
        self.retryable = getattr(error, "retryable", False) is True
        super().__init__(
            f"policy inference failed for request {request.serial}: {error}")


@dataclass
class CommandTicket:
    pose: np.ndarray
    gripper: float | None
    submitted_time: float
    done: threading.Event
    started_time: float | None = None
    completed_time: float | None = None
    command_pose: np.ndarray | None = None
    command_grippers: tuple[float, float] | None = None
    error: BaseException | None = None


class CommandRPCWorker:
    """Keep one ordered HAL RPC in flight while the servo loop sleeps."""
    def __init__(self, hal):
        self.hal = hal
        self.requests = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.active = None
        self.thread = threading.Thread(
            target=self._run, name="xr1-hal-command", daemon=True)
        self.closed = False
        self.thread.start()

    def _run(self):
        while True:
            ticket = self.requests.get()
            if ticket is None:
                return
            ticket.started_time = time.monotonic()
            try:
                ticket.command_pose, ticket.command_grippers = self.hal.send_command(
                    ticket.pose, ticket.gripper)
            except BaseException as error:
                ticket.error = error
            finally:
                ticket.completed_time = time.monotonic()
                ticket.done.set()

    def submit(self, pose, gripper):
        ticket = CommandTicket(
            np.asarray(pose, dtype=np.float64).copy(), gripper,
            time.monotonic(), threading.Event())
        with self.lock:
            if self.closed:
                raise HardwareError("HAL command worker is closed")
            if self.active is not None:
                raise HardwareError("HAL command worker has an unresolved command")
            self.active = ticket
            self.requests.put_nowait(ticket)
        return ticket

    def wait(self, ticket, timeout):
        if not ticket.done.wait(timeout):
            raise HardwareError(f"HAL command RPC did not return within {timeout:.3f}s")
        with self.lock:
            if self.active is ticket:
                self.active = None
        if ticket.error is not None:
            raise HardwareError("HAL command RPC failed") from ticket.error
        return ticket

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.requests.put(None)
        self.thread.join(2)
        if self.thread.is_alive():
            LOG.error("HAL command worker did not stop within 2s")


class Predictor:
    def __init__(self, policy):
        self.policy = policy
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name="xr1-policy", daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            request = self.requests.get()
            if request is None:
                return
            started = time.monotonic()
            try:
                chunk, metadata = self.policy.request(request.raw, request.serial, request.prefix)
                self.results.put((request, chunk, metadata, time.monotonic()-started))
            except BaseException as error:
                self.results.put(PolicyInferenceError(request, error))

    def submit(self, request):
        self.requests.put_nowait(request)

    def receive(self, timeout):
        result = self.results.get(timeout=timeout)
        if isinstance(result, BaseException):
            raise result from result.original_error
        return result

    def close(self):
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.policy.close()
        self.thread.join(2)


def encode_prefix(raw, poses, grippers):
    """Re-express committed absolute targets relative to the new observation."""
    poses = np.asarray(poses, dtype=np.float64)
    grippers = np.asarray(grippers, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] != len(grippers):
        raise ValueError("invalid committed Cartesian prefix")
    observed = np.asarray(raw["state"]["left_cart"], dtype=np.float64)
    observed_rotation = quaternion_matrix(observed[3:])
    action = np.zeros((len(poses), 60), dtype=np.float32)
    action[:, :3] = (poses[:, :3] - observed[:3]) @ observed_rotation
    action[:, 3:6] = np.stack([relative_rotvec(observed[3:], pose[3:]) for pose in poses])
    action[:, 6] = grippers - float(raw["state"]["left_gripper"][0])
    return action


def pose_error(actual, target):
    return float(np.linalg.norm(actual[:3]-target[:3])), quaternion_angle(actual[3:], target[3:])


def _format_optional(value, scale_or_transform=1.0):
    if value is None:
        return "unavailable"
    transformed = (scale_or_transform(value) if callable(scale_or_transform)
                   else value*scale_or_transform)
    return f"{transformed:.2f}"


def validate_prefix_handover(request, chunk):
    if request.prefix is None:
        return None
    length = len(request.prefix)
    position = np.linalg.norm(chunk["left_pose"][:length,:3]-request.committed_pose[:,:3], axis=1).max()
    angle = max(quaternion_angle(a[3:], b[3:])
                for a,b in zip(chunk["left_pose"][:length], request.committed_pose))
    gripper = np.abs(chunk["left_gripper"][:length]-request.committed_gripper).max()
    if position > .001 or angle > np.deg2rad(1) or gripper > .001:
        raise HardwareError(f"conditioned prefix mismatch: {position:.4f}m {np.rad2deg(angle):.2f}deg {gripper:.4f}m")
    return {"position_error_m": float(position),
            "rotation_error_rad": float(angle),
            "gripper_error_m": float(gripper)}


def policy_rejection_details(error):
    metadata = dict(getattr(error, "metadata", {}))
    keys = (
        "error", "error_code", "retryable", "boundary_kind", "frame_index",
        "frame_number", "boundary_value", "boundary_limit", "observed", "target",
        "rejected_prediction_path",
    )
    return {key: metadata[key] for key in keys if key in metadata}


class Runner:
    def __init__(self, args, observations, hal, predictor):
        self.args, self.observations, self.hal, self.predictor = args, observations, hal, predictor
        self.serial = 0
        self.inferences = 0
        self.command_worker = None
        self.pending_command = None
        self.consecutive_command_rpc_timeouts = 0
        self.trace = [{"event": "run_config", "time": time.monotonic(),
                       "wall_time": datetime.now().isoformat(),
                       "args": vars(args).copy(), "model_frequency_hz": 30,
                       "control_hz": args.control_hz,
                       "policy_hz": args.control_hz,
                       "servo_hz": args.servo_hz,
                       "arm_command_space": "cartesian",
                       "interpolation": "c2_quintic_stationary_chunk_boundary",
                       "trajectory_time_scale": 1,
                       "hal_command_rpc": ("single_inflight_pipelined"
                           if args.inference_mode == "async" else "synchronous"),
                       "prefix_window_s": args.prefix_length / args.control_hz,
                       "async_latency_margin_s": args.async_latency_margin_s,
                       "max_async_handover_late_s": args.max_async_handover_late_s,
                       "async_trajectory_retries": args.async_trajectory_retries}]

    def _start_command_worker(self):
        if self.args.inference_mode == "async" and self.command_worker is None:
            self.command_worker = CommandRPCWorker(self.hal)

    def _finish_pending_command(self, enforce_limits=True):
        if self.pending_command is None:
            return
        ticket, row = self.pending_command
        wait_started = time.monotonic()
        ticket = self.command_worker.wait(ticket, self.args.request_timeout)
        row["command_wait_s"] = time.monotonic()-wait_started
        self.pending_command = None
        command_rpc_s = ticket.completed_time-ticket.started_time
        row.update({
            "time": ticket.completed_time,
            "command_rpc_return_time": ticket.completed_time,
            "pose_rpc_return_time": ticket.completed_time,
            "gripper_rpc_return_time": (ticket.completed_time
                if row["gripper_command_sent"] else None),
            "pose_rpc_s": command_rpc_s,
            "command_rpc_s": command_rpc_s,
            "command_pending": False,
            "command": {
                "leftarm_cmd_cart_pos": ticket.command_pose.tolist(),
                "leftarm_gripper_cmd_pos": (None if ticket.command_grippers is None
                                               else ticket.command_grippers[0]),
                "rightarm_gripper_cmd_pos": (None if ticket.command_grippers is None
                                                else ticket.command_grippers[1]),
            },
        })
        if command_rpc_s > self.args.dispatch_rebase_s:
            self.trace.append({"event":"command_rpc_delay", "serial":row["serial"],
                               "index":row["index"],
                               "stream_index":row["stream_index"],
                               "time":ticket.completed_time,
                               "duration_s":command_rpc_s,
                               "control_thread_blocked_s":
                                   row.get("command_wait_s", 0.0)})
            LOG.warning("HAL command RPC took %.3fs at stream frame %d "
                        "(policy frame %d); pipelined wait %.3fs",
                        command_rpc_s, row["stream_index"], row["index"],
                        row.get("command_wait_s", 0.0))
        if not enforce_limits:
            return
        if command_rpc_s > self.args.max_dispatch_late_s:
            self.consecutive_command_rpc_timeouts += 1
            hard_limit_s = 2*self.args.max_dispatch_late_s
            hard_timeout = command_rpc_s > hard_limit_s
            tolerated = (not hard_timeout and
                         self.consecutive_command_rpc_timeouts < 2)
            self.trace.append({"event":"command_rpc_timeout",
                               "serial":row["serial"], "index":row["index"],
                               "stream_index":row["stream_index"],
                               "time":ticket.completed_time,
                               "duration_s":command_rpc_s,
                               "limit_s":self.args.max_dispatch_late_s,
                               "hard_limit_s":hard_limit_s,
                               "consecutive":self.consecutive_command_rpc_timeouts,
                               "tolerated":tolerated})
            if not tolerated:
                qualifier = "hard" if hard_timeout else "consecutive"
                raise HardwareError(
                    f"HAL command RPC {qualifier} timeout: "
                    f"{command_rpc_s:.3f}s (limit "
                    f"{self.args.max_dispatch_late_s:.3f}s, consecutive "
                    f"{self.consecutive_command_rpc_timeouts})")
            LOG.warning(
                "tolerating one HAL command RPC overrun %.3fs/%.3fs at "
                "stream frame %d; the next overrun will stop control",
                command_rpc_s, self.args.max_dispatch_late_s,
                row["stream_index"])
        else:
            self.consecutive_command_rpc_timeouts = 0

    def close(self):
        if self.pending_command is not None:
            try:
                self._finish_pending_command(enforce_limits=False)
            except Exception:
                LOG.exception("failed to drain final HAL command RPC")
        if self.command_worker is not None:
            self.command_worker.close()
            self.command_worker = None

    def feedback(self):
        with_metadata = getattr(self.hal, "feedback_with_metadata", None)
        if with_metadata is not None:
            pose, joints, left_grip, right_grip, stamp, sequence, rpc_s = with_metadata()
            consume_recovery = getattr(self.hal, "consume_feedback_recovery", None)
            recovery = consume_recovery() if consume_recovery is not None else None
            if recovery is not None:
                self.trace.append({"event": "feedback_recovered",
                                   "serial": self.serial,
                                   "time": time.monotonic(), **recovery})
            return (pose, joints, left_grip, right_grip, stamp, sequence, rpc_s,
                    "hal_rpc_background")
        pose, joints, left_grip, right_grip = self.hal.feedback()
        return pose, joints, left_grip, right_grip, None, None, None, "hal_test_fallback"

    def observation(self):
        state, stamp, sequence = self.observations.latest()
        raw = build_raw(state, self.args.task)
        return raw, stamp, sequence

    def submit(self, prefix_pose=None, prefix_gripper=None):
        raw, stamp, sequence = self.observation()
        prefix = None if prefix_pose is None else encode_prefix(raw, prefix_pose, prefix_gripper)
        self.serial += 1
        request = Request(self.serial, raw, prefix, prefix_pose, prefix_gripper)
        self.predictor.submit(request)
        self.trace.append({"event":"request", "serial":self.serial, "time":time.monotonic(),
                           "observation_stamp":stamp, "sequence":sequence,
                           "prefix_length":0 if prefix is None else len(prefix)})
        return request

    def initial(self):
        request = self.submit()
        result = self.predictor.receive(self.args.request_timeout)
        self.inferences += 1
        self.trace.append({"event": "initial_result", "serial": request.serial,
                           "time": time.monotonic(), "latency_s": result[3],
                           "server_latency_s": result[2].get("latency_s"),
                           "model_latency_s": result[2].get("model_latency_s"),
                           "postprocess_latency_s":
                               result[2].get("postprocess_latency_s"),
                           "client_send_s": result[2].get("client_send_s"),
                           "client_wait_and_receive_s":
                               result[2].get("client_wait_and_receive_s"),
                           "request_bytes": result[2].get("request_bytes")})
        return result

    def initial_for_execution(self):
        """Discard bounded cold-start results while the robot is stationary."""
        result = self.initial()
        for attempt in range(1, self.args.async_startup_retries + 1):
            want_async_continuation = (self.args.inference_mode == "async" and
                (self.args.chunks == 0 or self.inferences < self.args.chunks))
            latency_limit_s = (self.args.prefix_length / self.args.control_hz -
                               self.args.async_latency_margin_s)
            if not want_async_continuation or result[3] <= latency_limit_s:
                break
            request = result[0]
            self.trace.append({"event": "async_startup_retry", "serial": request.serial,
                               "time": time.monotonic(), "attempt": attempt,
                               "max_retries": self.args.async_startup_retries,
                               "discarded_latency_s": result[3],
                               "latency_limit_s": latency_limit_s})
            LOG.warning(
                "discarding stationary async cold-start result %d: %.3fs exceeds "
                "%.3fs budget; retrying (%d/%d)",
                request.serial, result[3], latency_limit_s, attempt,
                self.args.async_startup_retries)
            # A discarded warm-up is not one of the chunks requested by the
            # operator.  Capture a fresh observation for the replacement.
            self.inferences -= 1
            result = self.initial()
        return result

    def _recover_async_prediction(self, failed_request, failure, target,
                                  target_gripper, period):
        """Soft-stop, then retry a rejected async trajectory from fresh state."""
        recovery_started = time.monotonic()
        details = policy_rejection_details(failure)
        self.trace.append({
            "event": "async_prediction_rejected", "serial": failed_request.serial,
            "time": time.monotonic(), "stage": "prefetch", **details})
        if not failure.retryable or self.args.async_trajectory_retries < 1:
            raise failure

        LOG.warning(
            "async request %d rejected by %s; holding the final safe target and "
            "retrying from a fresh stationary observation (%d allowed): %s",
            failed_request.serial, failure.error_code or "policy_validation",
            self.args.async_trajectory_retries, failure)
        # The robot may still lag the last commanded target by a few millimetres.
        # Freeze around the latest measured state instead of asking a stopped
        # controller to continue toward the rejected handover boundary.
        hold_pose, _, hold_gripper, *_ = self.feedback()
        self.trace.append({
            "event": "async_recovery_hold", "serial": failed_request.serial,
            "time": time.monotonic(), "hold_pose": hold_pose.tolist(),
            "hold_gripper": float(hold_gripper),
            "intended_final_pose": np.asarray(target).tolist(),
            "intended_final_gripper": float(target_gripper)})
        tracking_stopped = self.wait_settled(hold_pose, float(hold_gripper))
        if not tracking_stopped:
            raise HardwareError(
                "cannot retry a rejected asynchronous trajectory without "
                "soft-stopping ONLINE_TRACKING") from failure

        last_failure = failure
        for attempt in range(1, self.args.async_trajectory_retries + 1):
            retry_request = self.submit()
            retry_started = time.monotonic()
            self.trace.append({
                "event": "async_recovery_retry", "serial": retry_request.serial,
                "replaces_serial": failed_request.serial,
                "time": retry_started, "attempt": attempt,
                "max_retries": self.args.async_trajectory_retries})
            try:
                result = self.predictor.receive(self.args.request_timeout)
            except queue.Empty as error:
                self.trace.append({
                    "event": "async_recovery_failed", "serial": retry_request.serial,
                    "time": time.monotonic(), "attempt": attempt,
                    "reason": "inference_timeout"})
                raise HardwareError(
                    f"stationary recovery inference {retry_request.serial} timed out") \
                    from error
            except PolicyInferenceError as error:
                last_failure = error
                retry_details = policy_rejection_details(error)
                self.trace.append({
                    "event": "async_prediction_rejected",
                    "serial": retry_request.serial, "time": time.monotonic(),
                    "stage": "stationary_retry", "attempt": attempt,
                    **retry_details})
                if (error.request is retry_request and error.retryable and
                        attempt < self.args.async_trajectory_retries):
                    LOG.warning(
                        "stationary recovery request %d was also rejected; "
                        "retrying (%d/%d): %s", retry_request.serial, attempt,
                        self.args.async_trajectory_retries, error)
                    continue
                self.trace.append({
                    "event": "async_recovery_failed", "serial": retry_request.serial,
                    "time": time.monotonic(), "attempt": attempt,
                    "reason": "policy_rejected", **retry_details})
                raise HardwareError(
                    f"stationary recovery inference {retry_request.serial} failed: "
                    f"{error}") from error

            request, chunk, metadata, latency = result
            if request is not retry_request or request.prefix is not None:
                raise HardwareError("stationary recovery response/request mismatch")
            current, *_ = self.feedback()
            drift = pose_error(current, chunk["observation_pose"])
            if (drift[0] > self.args.max_observation_drift_m or
                    drift[1] > self.args.max_observation_drift_rad):
                raise HardwareError(
                    f"robot moved during stationary recovery inference: "
                    f"{drift[0]:.3f}m {np.rad2deg(drift[1]):.1f}deg")
            self.hal.start_tracking()
            self.inferences += 1
            resumed = time.monotonic()
            self.trace.append({
                "event": "async_recovery_handover", "serial": request.serial,
                "replaces_serial": failed_request.serial, "time": resumed,
                "attempt": attempt, "latency_s": latency,
                "server_latency_s": metadata.get("latency_s"),
                "model_latency_s": metadata.get("model_latency_s"),
                "postprocess_latency_s": metadata.get("postprocess_latency_s"),
                "client_send_s": metadata.get("client_send_s"),
                "client_wait_and_receive_s":
                    metadata.get("client_wait_and_receive_s"),
                "request_bytes": metadata.get("request_bytes"),
                "position_drift_m": drift[0],
                "rotation_drift_rad": drift[1],
                "retry_elapsed_s": resumed-retry_started,
                "recovery_pause_s": resumed-recovery_started})
            return result, current, resumed + period

        # The loop can only fall through if the configured retry count changes
        # concurrently, which argparse and normal execution do not permit.
        raise HardwareError("asynchronous trajectory recovery exhausted") from last_failure

    def execute(self, first_result):
        request, chunk, metadata, latency = first_result
        validate_prefix_handover(request, chunk)
        want_async_continuation = (self.args.inference_mode == "async" and
            (self.args.chunks == 0 or self.inferences < self.args.chunks))
        if want_async_continuation:
            prefix_window_s = self.args.prefix_length/self.args.control_hz
            required_window_s = latency + self.args.async_latency_margin_s
            budget = {"event": "async_budget_check", "serial": request.serial,
                      "time": time.monotonic(), "initial_latency_s": latency,
                      "latency_margin_s": self.args.async_latency_margin_s,
                      "required_window_s": required_window_s,
                      "prefix_window_s": prefix_window_s,
                      "slack_s": prefix_window_s-required_window_s,
                      "accepted": required_window_s <= prefix_window_s}
            self.trace.append(budget)
            if not budget["accepted"]:
                raise HardwareError(
                    f"asynchronous prefix window is too short: "
                    f"{prefix_window_s:.3f}s < initial inference {latency:.3f}s + "
                    f"margin {self.args.async_latency_margin_s:.3f}s")
        # Initial inference happens while stationary. Refuse to start if another
        # controller moved the robot after the observation.
        current, *_ = self.feedback()
        drift = pose_error(current, chunk["observation_pose"])
        if drift[0] > self.args.max_observation_drift_m or drift[1] > self.args.max_observation_drift_rad:
            raise HardwareError(f"robot moved during initial inference: {drift[0]:.3f}m {np.rad2deg(drift[1]):.1f}deg")
        self.hal.start_tracking()
        self._start_command_worker()

        period = 1 / self.args.servo_hz
        trajectory_anchor = current.copy()
        # start_tracking() seeds Cartesian mode with the measured pose.  Treat
        # that seed like every other command and leave one full client period
        # before the first policy target.
        next_dispatch_not_before = time.monotonic() + period
        while True:
            prefix_length = 0 if request.prefix is None else len(request.prefix)
            poses = chunk["left_pose"][prefix_length:]
            grippers = chunk["left_gripper"][prefix_length:]
            if not len(poses):
                raise HardwareError("model returned no unexecuted Cartesian targets")
            dense = interpolate_cartesian_targets(
                trajectory_anchor, poses, self.args.control_hz, self.args.servo_hz)
            dense_with_anchor = np.vstack([trajectory_anchor, dense["pose"]])
            position_steps = np.linalg.norm(np.diff(dense_with_anchor[:, :3], axis=0), axis=1)
            rotation_steps = np.asarray([
                quaternion_angle(left[3:], right[3:])
                for left, right in zip(dense_with_anchor, dense_with_anchor[1:])
            ])
            max_position_step = float(position_steps.max())
            max_rotation_step = float(rotation_steps.max())
            cap_linear = getattr(self.hal, "cap_linear", None)
            cap_angular = getattr(self.hal, "cap_angular", None)
            if cap_linear is not None and max_position_step > cap_linear/self.args.servo_hz:
                raise HardwareError(
                    f"interpolated Cartesian linear speed exceeds HAL capability: "
                    f"{max_position_step*self.args.servo_hz:.3f}m/s > {cap_linear:.3f}m/s")
            if cap_angular is not None and max_rotation_step > cap_angular/self.args.servo_hz:
                raise HardwareError(
                    f"interpolated Cartesian angular speed exceeds HAL capability: "
                    f"{max_rotation_step*self.args.servo_hz:.3f}rad/s > {cap_angular:.3f}rad/s")
            self.trace.append({
                "event": "trajectory_plan", "serial": request.serial,
                "time": time.monotonic(), "policy_frames": int(len(poses)),
                "stream_frames": int(len(dense["pose"])),
                "policy_hz": self.args.control_hz, "servo_hz": self.args.servo_hz,
                "duration_s": len(poses)/self.args.control_hz,
                "trajectory_time_scale": 1,
                "interpolation": dense["spline"],
                "interpolation_steps": dense["interpolation_steps"],
                "boundary_velocity_zero": dense["boundary_velocity_zero"],
                "boundary_acceleration_zero": dense["boundary_acceleration_zero"],
                "gripper_interpolated": False,
                "gripper_dispatch_hz": self.args.control_hz,
                "max_position_step_m": max_position_step,
                "max_rotation_step_rad": max_rotation_step,
            })
            want_next = self.args.chunks == 0 or self.inferences < self.args.chunks
            prefetch_at = len(poses) - self.args.prefix_length
            next_request = None
            async_prefetch_started = None
            async_planned_handover = None
            previous_target = trajectory_anchor
            for stream_index, (pose, policy_index, interpolation_step) in enumerate(zip(
                    dense["pose"], dense["policy_index"], dense["interpolation_step"])):
                policy_index = int(policy_index)
                interpolation_step = int(interpolation_step)
                policy_frame = policy_index + prefix_length
                policy_pose = poses[policy_index]
                gripper = float(grippers[policy_index])
                is_policy_keyframe = interpolation_step == dense["interpolation_steps"]
                scheduled_time = next_dispatch_not_before
                wake_time = time.monotonic()
                remaining = scheduled_time - wake_time
                if remaining > 0:
                    time.sleep(remaining)
                elif -remaining > self.args.max_dispatch_late_s:
                    self.trace.append({"event":"dispatch_timeout", "serial":request.serial,
                                       "index":policy_frame, "stream_index":stream_index,
                                       "time":wake_time,
                                       "phase":"wake", "scheduled_time":scheduled_time,
                                       "late_s":-remaining})
                    raise HardwareError(f"Cartesian dispatch late by {-remaining:.3f}s")
                # The previous command RPC normally completes during this
                # frame's servo sleep. Only its excess beyond one period can
                # block the control thread; commands remain strictly ordered.
                self._finish_pending_command()
                wake_time = time.monotonic()

                if (self.args.inference_mode == "async" and
                        next_request is None and policy_index == prefetch_at and
                        interpolation_step == 1 and want_next):
                    committed_pose = poses[policy_index:policy_index+self.args.prefix_length].copy()
                    committed_gripper = grippers[
                        policy_index:policy_index+self.args.prefix_length].copy()
                    async_prefetch_started = time.monotonic()
                    next_request = self.submit(committed_pose, committed_gripper)
                    frames_remaining = len(dense["pose"])-stream_index
                    async_planned_handover = scheduled_time + frames_remaining*period
                    self.trace.append({
                        "event": "async_prefetch", "serial": request.serial,
                        "next_serial": next_request.serial,
                        "time": time.monotonic(),
                        "policy_index": policy_frame,
                        "stream_index": stream_index,
                        "prefix_length": len(committed_pose),
                        "prefix_window_s": len(committed_pose)/self.args.control_hz,
                        "planned_handover_time": async_planned_handover,
                        "committed_first_pose": committed_pose[0].tolist(),
                        "committed_last_pose": committed_pose[-1].tolist(),
                        "committed_first_gripper": float(committed_gripper[0]),
                        "committed_last_gripper": float(committed_gripper[-1]),
                    })

                feedback_started = time.monotonic()
                try:
                    (actual, joints, actual_grip, _, feedback_stamp,
                     feedback_sequence, feedback_poll_rpc_s,
                     feedback_source) = self.feedback()
                except BaseException as error:
                    self.trace.append({"event":"feedback_error", "serial":request.serial,
                                       "index":policy_frame, "stream_index":stream_index,
                                       "time":time.monotonic(),
                                       "scheduled_time":scheduled_time,
                                       "feedback_started_time":feedback_started,
                                       "type":type(error).__name__, "message":str(error)})
                    raise
                feedback_time = time.monotonic()
                if previous_target is not None:
                    follow = pose_error(actual, previous_target)
                    if follow[0] > self.args.max_follow_error_m or follow[1] > self.args.max_follow_error_rad:
                        raise HardwareError(f"Cartesian following error: {follow[0]:.3f}m {np.rad2deg(follow[1]):.1f}deg")

                # Account for time spent obtaining feedback (and, in async mode,
                # submitting the next observation) before dispatch.  A bounded
                # non-real-time scheduling stall should stretch the remainder of
                # the chunk instead of causing catch-up bursts.  Larger stalls
                # still fail closed before a stale target is sent.
                dispatch_time = time.monotonic()
                dispatch_late = max(0.0, dispatch_time-scheduled_time)
                if dispatch_late > self.args.max_dispatch_late_s:
                    self.trace.append({"event":"dispatch_timeout", "serial":request.serial,
                                       "index":policy_frame, "stream_index":stream_index,
                                       "time":dispatch_time,
                                       "phase":"feedback", "scheduled_time":scheduled_time,
                                       "feedback_started_time":feedback_started,
                                       "feedback_return_time":feedback_time,
                                       "feedback_source_time":feedback_stamp,
                                       "feedback_sequence":feedback_sequence,
                                       "feedback_poll_rpc_s":feedback_poll_rpc_s,
                                       "late_s":dispatch_late})
                    raise HardwareError(f"Cartesian dispatch late by {dispatch_late:.3f}s")
                if dispatch_late > self.args.dispatch_rebase_s:
                    self.trace.append({"event":"dispatch_rebase", "serial":request.serial,
                                       "index":policy_frame, "stream_index":stream_index,
                                       "time":dispatch_time, "late_s":dispatch_late})
                    LOG.warning("Cartesian dispatch delayed %.3fs at stream frame %d "
                                "(policy frame %d); rebasing cadence",
                                dispatch_late, stream_index, policy_frame)

                gripper_command_sent = (is_policy_keyframe and
                    abs(gripper-actual_grip) > self.args.gripper_deadband_m)
                pose_rpc_started = time.monotonic()
                target_row = {"event":"target", "serial":request.serial,
                              "index":policy_frame, "stream_index":stream_index,
                              "interpolation_step":interpolation_step,
                              "interpolation_steps":dense["interpolation_steps"],
                              "is_policy_keyframe":is_policy_keyframe,
                              "time":pose_rpc_started, "scheduled_time":scheduled_time,
                              "wake_time":wake_time,
                              "wake_late_s":max(0.0, wake_time-scheduled_time),
                              "feedback_time":feedback_time,
                              "feedback_started_time":feedback_started,
                              "feedback_return_time":feedback_time,
                              "feedback_call_s":feedback_time-feedback_started,
                              "feedback_source":feedback_source,
                              "feedback_source_time":feedback_stamp,
                              "feedback_sequence":feedback_sequence,
                              "feedback_poll_rpc_s":feedback_poll_rpc_s,
                              "feedback_age_s": (None if feedback_stamp is None
                                                   else feedback_time-feedback_stamp),
                              "dispatch_started_time":pose_rpc_started,
                              "command_rpc_return_time":None,
                              "pose_rpc_return_time":None,
                              "gripper_rpc_return_time":None,
                              "pose_rpc_s":None, "command_rpc_s":None,
                              "command_rpc_count":1,
                              "command_wait_s":0.0,
                              "policy_pose":policy_pose.tolist(),
                              "pose":pose.tolist(), "actual":actual.tolist(),
                              "joints":joints.tolist(), "policy_gripper":gripper,
                              "gripper":gripper,
                              "actual_gripper":actual_grip,
                              "command": {
                                  "leftarm_cmd_cart_pos":np.asarray(pose).tolist(),
                                  "leftarm_gripper_cmd_pos":(
                                      gripper if gripper_command_sent else None),
                                  "rightarm_gripper_cmd_pos":None,
                              },
                              "gripper_command_sent":gripper_command_sent,
                              "dispatch_late_s":dispatch_late,
                              "command_pending":self.command_worker is not None}
                if self.command_worker is not None:
                    ticket = self.command_worker.submit(
                        pose, gripper if gripper_command_sent else None)
                    next_dispatch_not_before = pose_rpc_started + period
                    target_row["next_dispatch_not_before"] = next_dispatch_not_before
                    self.trace.append(target_row)
                    self.pending_command = (ticket, target_row)
                else:
                    command_pose, command_grippers = self.hal.send_command(
                        pose, gripper if gripper_command_sent else None)
                    command_completed = time.monotonic()
                    next_dispatch_not_before = command_completed + period
                    command_rpc_s = command_completed-pose_rpc_started
                    target_row.update({
                        "time":command_completed,
                        "command_rpc_return_time":command_completed,
                        "pose_rpc_return_time":command_completed,
                        "gripper_rpc_return_time":(
                            command_completed if gripper_command_sent else None),
                        "next_dispatch_not_before":next_dispatch_not_before,
                        "pose_rpc_s":command_rpc_s,
                        "command_rpc_s":command_rpc_s,
                        "command_pending":False,
                        "command": {
                            "leftarm_cmd_cart_pos":np.asarray(command_pose).tolist(),
                            "leftarm_gripper_cmd_pos":(
                                None if command_grippers is None else command_grippers[0]),
                            "rightarm_gripper_cmd_pos":(
                                None if command_grippers is None else command_grippers[1]),
                        },
                    })
                    self.trace.append(target_row)
                    if command_rpc_s > self.args.dispatch_rebase_s:
                        self.trace.append({"event":"command_rpc_delay",
                                           "serial":request.serial,
                                           "index":policy_frame,
                                           "stream_index":stream_index,
                                           "time":command_completed,
                                           "duration_s":command_rpc_s})
                        LOG.warning("HAL command RPC took %.3fs at stream frame %d "
                                    "(policy frame %d); preserving cadence",
                                    command_rpc_s, stream_index, policy_frame)
                    if command_rpc_s > self.args.max_dispatch_late_s:
                        self.consecutive_command_rpc_timeouts += 1
                        hard_limit_s = 2*self.args.max_dispatch_late_s
                        hard_timeout = command_rpc_s > hard_limit_s
                        tolerated = (not hard_timeout and
                                     self.consecutive_command_rpc_timeouts < 2)
                        self.trace.append({"event":"command_rpc_timeout",
                                           "serial":request.serial,
                                           "index":policy_frame,
                                           "stream_index":stream_index,
                                           "time":command_completed,
                                           "duration_s":command_rpc_s,
                                           "limit_s":self.args.max_dispatch_late_s,
                                           "hard_limit_s":hard_limit_s,
                                           "consecutive":
                                               self.consecutive_command_rpc_timeouts,
                                           "tolerated":tolerated})
                        if not tolerated:
                            qualifier = "hard" if hard_timeout else "consecutive"
                            raise HardwareError(
                                f"HAL command RPC {qualifier} timeout: "
                                f"{command_rpc_s:.3f}s (limit "
                                f"{self.args.max_dispatch_late_s:.3f}s, consecutive "
                                f"{self.consecutive_command_rpc_timeouts})")
                    else:
                        self.consecutive_command_rpc_timeouts = 0
                previous_target = pose

            self._finish_pending_command()
            trajectory_anchor = dense["pose"][-1].copy()

            if not want_next:
                self.wait_settled(poses[-1], float(grippers[-1]))
                return
            if self.args.inference_mode == "sync":
                # Complete the full official 30-frame chunk, soft-stop Cartesian
                # tracking, verify physical stability, then infer the next chunk.
                tracking_stopped = self.wait_settled(
                    poses[-1], float(grippers[-1]))
                next_request = self.submit()
                try:
                    next_result = self.predictor.receive(self.args.request_timeout)
                except queue.Empty as error:
                    raise HardwareError("synchronous inference timed out") from error
                request, chunk, metadata, latency = next_result
                if request is not next_request or request.prefix is not None:
                    raise HardwareError("synchronous response/request mismatch")
                current, *_ = self.feedback()
                drift = pose_error(current, chunk["observation_pose"])
                if (drift[0] > self.args.max_observation_drift_m or
                        drift[1] > self.args.max_observation_drift_rad):
                    raise HardwareError(
                        f"robot moved during synchronous inference: {drift[0]:.3f}m "
                        f"{np.rad2deg(drift[1]):.1f}deg")
                if tracking_stopped:
                    restart_started = time.monotonic()
                    self.trace.append({"event": "tracking_restart_start",
                                       "serial": request.serial,
                                       "time": restart_started})
                    self.hal.start_tracking()
                    self.trace.append({"event": "tracking_restart_complete",
                                       "serial": request.serial,
                                       "time": time.monotonic(),
                                       "duration_s": time.monotonic()-restart_started})
                self.inferences += 1
                self.trace.append({"event":"sync_handover", "serial":request.serial,
                                   "time":time.monotonic(), "latency_s":latency})
                trajectory_anchor = current.copy()
                # Begin the first interpolated sub-frame one servo period after
                # the measured anchor; never jump to it immediately after inference.
                next_dispatch_not_before = time.monotonic() + period
                continue
            if next_request is None:
                raise HardwareError("asynchronous request was not scheduled")
            # N committed actions occupy N/control_hz seconds, including the hold after
            # the last target. A late policy response cannot be joined faithfully.
            handover_deadline = next_dispatch_not_before
            timeout = max(0, handover_deadline-time.monotonic())
            try:
                next_result = self.predictor.receive(timeout)
            except queue.Empty as error:
                self.trace.append({"event": "async_handover_timeout",
                                   "serial": next_request.serial,
                                   "time": time.monotonic(),
                                   "reason": "inference_result_missing",
                                   "handover_deadline": handover_deadline,
                                   "prefix_window_s":
                                       self.args.prefix_length/self.args.control_hz})
                raise HardwareError(
                    f"inference exceeded {self.args.prefix_length}-frame prefix window "
                    f"({self.args.prefix_length/self.args.control_hz:.3f}s at "
                    f"{self.args.control_hz:g}Hz, request {next_request.serial})") from error
            except PolicyInferenceError as error:
                if error.request is not next_request:
                    raise HardwareError(
                        "asynchronous failed-response/request mismatch") from error
                if not error.retryable or self.args.async_trajectory_retries < 1:
                    self.trace.append({
                        "event": "async_prediction_rejected",
                        "serial": next_request.serial,
                        "time": time.monotonic(), "stage": "prefetch",
                        **policy_rejection_details(error)})
                    raise
                recovered, trajectory_anchor, next_dispatch_not_before = (
                    self._recover_async_prediction(
                        next_request, error, poses[-1], float(grippers[-1]), period))
                request, chunk, metadata, latency = recovered
                continue
            result_received_time = time.monotonic()
            request, chunk, metadata, latency = next_result
            if request is not next_request:
                raise HardwareError("asynchronous response/request mismatch")
            prefix_error = validate_prefix_handover(request, chunk)
            validated_time = time.monotonic()
            handover_late_s = max(0.0, validated_time-handover_deadline)
            if handover_late_s > self.args.max_async_handover_late_s:
                self.trace.append({"event": "async_handover_timeout",
                                   "serial": request.serial,
                                   "time": validated_time,
                                   "reason": "validated_result_late",
                                   "handover_deadline": handover_deadline,
                                   "late_s": handover_late_s,
                                   "limit_s": self.args.max_async_handover_late_s,
                                   "latency_s": latency})
                raise HardwareError(
                    f"asynchronous handover late by {handover_late_s:.3f}s "
                    f"(limit {self.args.max_async_handover_late_s:.3f}s)")
            remaining = handover_deadline-validated_time
            if remaining > 0:
                time.sleep(remaining)
            self.inferences += 1
            self.trace.append({"event":"handover", "serial":request.serial,
                               "time":time.monotonic(), "mode":"async",
                               "latency_s":latency,
                               "server_latency_s":metadata.get("latency_s"),
                               "model_latency_s":metadata.get("model_latency_s"),
                               "postprocess_latency_s":
                                   metadata.get("postprocess_latency_s"),
                               "client_send_s":metadata.get("client_send_s"),
                               "client_wait_and_receive_s":
                                   metadata.get("client_wait_and_receive_s"),
                               "request_bytes":metadata.get("request_bytes"),
                               "discarded_prefix":len(request.prefix),
                               "prefetch_elapsed_s":
                                   time.monotonic()-async_prefetch_started,
                               "planned_handover_time":async_planned_handover,
                               "handover_deadline":handover_deadline,
                               "result_received_time":result_received_time,
                               "result_slack_s":handover_deadline-result_received_time,
                               "validation_s":validated_time-result_received_time,
                               "handover_late_s":handover_late_s,
                               "prefix_position_error_m":
                                   prefix_error["position_error_m"],
                               "prefix_rotation_error_rad":
                                   prefix_error["rotation_error_rad"],
                               "prefix_gripper_error_m":
                                   prefix_error["gripper_error_m"]})
            continue

    def wait_settled(self, target, target_gripper):
        target_gripper = float(target_gripper)
        if (not math.isfinite(target_gripper) or
                not -GRIPPER_COMMAND_TOLERANCE_M <= target_gripper <=
                INITIAL_GRIPPER_M+GRIPPER_COMMAND_TOLERANCE_M):
            raise HardwareError("settle gripper target outside 0..0.09 m")
        target_gripper = min(INITIAL_GRIPPER_M, max(0.0, target_gripper))
        tracking_stopped = False
        stop_tracking = getattr(self.hal, "stop_tracking", None)
        if stop_tracking is not None:
            stop_started = time.monotonic()
            self.trace.append({"event": "tracking_stop_start", "serial": self.serial,
                               "time": stop_started})
            try:
                tracking_stopped = bool(stop_tracking())
            except BaseException as error:
                self.trace.append({"event": "tracking_stop_error",
                                   "serial": self.serial, "time": time.monotonic(),
                                   "duration_s": time.monotonic()-stop_started,
                                   "type": type(error).__name__, "message": str(error)})
                raise
            self.trace.append({"event": "tracking_stop_complete",
                               "serial": self.serial, "time": time.monotonic(),
                               "duration_s": time.monotonic()-stop_started,
                               "tracking_stopped": tracking_stopped})
        started = time.monotonic()
        deadline = started + self.args.settle_timeout
        stable_since = None
        next_status = started
        last_sample = None
        previous_motion_sample = None
        previous_feedback_marker = None
        self.trace.append({"event": "settle_start", "serial": self.serial,
                           "time": started, "target": target.tolist(),
                           "target_gripper": target_gripper,
                           "timeout_s": self.args.settle_timeout,
                           "position_tolerance_m": self.args.settle_position_m,
                           "rotation_tolerance_rad": self.args.settle_rotation_rad,
                           "gripper_tolerance_m": self.args.settle_gripper_m,
                           "linear_speed_tolerance_m_s":
                               self.args.settle_linear_speed_m_s,
                           "angular_speed_tolerance_rad_s":
                               self.args.settle_angular_speed_rad_s,
                           "joint_speed_tolerance_deg_s":
                               self.args.settle_joint_speed_deg_s,
                           "required_stable_s": self.args.settle_stable_s,
                           "velocity_source": "finite difference of fresh HAL feedback",
                           "tracking_stopped_before_wait": tracking_stopped,
                           "target_updates_during_wait": False})
        while time.monotonic() < deadline:
            (actual, joints, left_grip, right_grip, feedback_stamp,
             feedback_sequence, feedback_poll_rpc_s, feedback_source) = self.feedback()
            now = time.monotonic()
            sample_time = (float(feedback_stamp) if feedback_stamp is not None
                           else now)
            feedback_marker = (("sequence", int(feedback_sequence))
                               if feedback_sequence is not None
                               else ("time", sample_time))
            # The control loop can read the same 60 Hz background sample twice.
            # Never let a duplicate advance the stable timer or manufacture a
            # zero velocity measurement.
            if feedback_marker == previous_feedback_marker:
                remaining = deadline-time.monotonic()
                if remaining > 0:
                    time.sleep(min(1/self.args.feedback_poll_hz, remaining))
                continue
            previous_feedback_marker = feedback_marker

            error = pose_error(actual, target)
            gripper_error = abs(left_grip-target_gripper)
            pose_within = (error[0] <= self.args.settle_position_m and
                           error[1] <= self.args.settle_rotation_rad)
            gripper_within = gripper_error <= self.args.settle_gripper_m

            velocity_dt = None
            linear_speed = None
            angular_speed = None
            joint_speed = None
            velocity_available = False
            if previous_motion_sample is not None:
                previous_time, previous_pose, previous_joints = previous_motion_sample
                velocity_dt = sample_time-previous_time
                if math.isfinite(velocity_dt) and velocity_dt > 0:
                    linear_speed = float(
                        np.linalg.norm(actual[:3]-previous_pose[:3]) / velocity_dt)
                    angular_speed = quaternion_angle(
                        previous_pose[3:], actual[3:]) / velocity_dt
                    joint_speed = float(np.max(
                        np.abs(joints-previous_joints)) / velocity_dt)
                    velocity_available = all(math.isfinite(value) for value in (
                        linear_speed, angular_speed, joint_speed))
            previous_motion_sample = (
                sample_time, actual.copy(), joints.copy())
            motion_within = (velocity_available and
                             linear_speed <= self.args.settle_linear_speed_m_s and
                             angular_speed <= self.args.settle_angular_speed_rad_s and
                             joint_speed <= self.args.settle_joint_speed_deg_s)
            within = pose_within and gripper_within and motion_within
            if within:
                if stable_since is None:
                    stable_since = sample_time
            else:
                stable_since = None
            stable_s = (0.0 if stable_since is None else
                        max(0.0, sample_time-stable_since))
            last_sample = {"event": "settle_feedback", "serial": self.serial,
                           "time": now, "elapsed_s": now-started,
                           "target": target.tolist(), "actual": actual.tolist(),
                           "joints": joints.tolist(), "target_gripper": target_gripper,
                           "actual_gripper": left_grip,
                           "right_gripper": right_grip,
                           "feedback_source": feedback_source,
                           "feedback_source_time": feedback_stamp,
                           "feedback_sequence": feedback_sequence,
                           "feedback_poll_rpc_s": feedback_poll_rpc_s,
                           "feedback_age_s": (None if feedback_stamp is None
                                                else now-feedback_stamp),
                           "position_error_m": error[0],
                           "rotation_error_rad": error[1],
                           "gripper_error_m": gripper_error,
                           "position_error_xyz_m": (target[:3]-actual[:3]).tolist(),
                           "velocity_sample_dt_s": velocity_dt,
                           "velocity_available": velocity_available,
                           "linear_speed_m_s": linear_speed,
                           "angular_speed_rad_s": angular_speed,
                           "joint_speed_deg_s": joint_speed,
                           "pose_within_tolerance": pose_within,
                           "gripper_within_tolerance": gripper_within,
                           "motion_within_tolerance": motion_within,
                           "within_tolerance": within, "stable_s": stable_s}
            self.trace.append(last_sample)
            if now >= next_status:
                diagnostics = getattr(self.hal, "diagnostics", None)
                status = diagnostics() if diagnostics else {"available": False}
                self.trace.append({"event": "settle_status", "serial": self.serial,
                                   "time": time.monotonic(), "status": status})
                LOG.info("settling request %d: %.2fs position=%.2fmm/%.2fmm "
                         "rotation=%.2fdeg/%.2fdeg gripper=%.2fmm/%.2fmm "
                         "speed=%s/%.1fmm/s angular_speed=%s/%.2fdeg/s "
                         "joint_speed=%s/%.2fdeg/s stable=%.2fs status=%s",
                         self.serial, now-started, error[0]*1000,
                         self.args.settle_position_m*1000, math.degrees(error[1]),
                         math.degrees(self.args.settle_rotation_rad), gripper_error*1000,
                         self.args.settle_gripper_m*1000,
                         "n/a" if linear_speed is None else f"{linear_speed*1000:.1f}",
                         self.args.settle_linear_speed_m_s*1000,
                         "n/a" if angular_speed is None else
                             f"{math.degrees(angular_speed):.2f}",
                         math.degrees(self.args.settle_angular_speed_rad_s),
                         "n/a" if joint_speed is None else f"{joint_speed:.2f}",
                         self.args.settle_joint_speed_deg_s, stable_s, status)
                next_status = time.monotonic() + .5
            if (stable_s >= self.args.settle_stable_s and
                    time.monotonic() <= deadline):
                self.trace.append({**last_sample, "event": "settle_complete"})
                return tracking_stopped
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(min(1/self.args.feedback_poll_hz, remaining))
        self.trace.append({**(last_sample or {}), "event": "settle_timeout",
                           "time": time.monotonic(), "serial": self.serial,
                           "elapsed_s": time.monotonic()-started})
        detail = "no feedback sample"
        if last_sample is not None:
            detail = (f"position={last_sample['position_error_m']*1000:.2f}mm "
                      f"(limit {self.args.settle_position_m*1000:.2f}mm), "
                      f"rotation={math.degrees(last_sample['rotation_error_rad']):.2f}deg "
                      f"(limit {math.degrees(self.args.settle_rotation_rad):.2f}deg), "
                      f"gripper={last_sample['gripper_error_m']*1000:.2f}mm "
                      f"(limit {self.args.settle_gripper_m*1000:.2f}mm), "
                      f"linear_speed={_format_optional(last_sample['linear_speed_m_s'], 1000)}mm/s "
                      f"(limit {self.args.settle_linear_speed_m_s*1000:.2f}mm/s), "
                      f"angular_speed={_format_optional(last_sample['angular_speed_rad_s'], math.degrees)}deg/s "
                      f"(limit {math.degrees(self.args.settle_angular_speed_rad_s):.2f}deg/s), "
                      f"joint_speed={_format_optional(last_sample['joint_speed_deg_s'])}deg/s "
                      f"(limit {self.args.settle_joint_speed_deg_s:.2f}deg/s), "
                      f"stable={last_sample['stable_s']:.3f}s "
                      f"(required {self.args.settle_stable_s:.3f}s), "
                      f"target={target.tolist()}, actual={last_sample['actual']}, "
                      f"target_gripper={target_gripper:.5f}, "
                      f"actual_gripper={last_sample['actual_gripper']:.5f}")
        raise HardwareError(
            f"final Cartesian/gripper target did not settle after "
            f"{self.args.settle_timeout:g}s: {detail}")

    def save_trace(self):
        path = ROOT / "runs/cartesian" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            for row in self.trace:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        LOG.info("trace saved to %s", path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hal-host", default="127.0.0.1")
    parser.add_argument("--hal-port", type=int, default=7860)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=10087)
    parser.add_argument("--task", default="Pick up the Ethernet cable and insert it into the RJ45 network port")
    parser.add_argument("--inference-mode", choices=("sync", "async"), default="async",
                        help="sync waits between full chunks; async uses official action_prefix")
    parser.add_argument("--chunks", type=int, default=0, help="number of model chunks; 0 runs continuously")
    parser.add_argument("--control-hz", type=float, default=15.0,
                        help="policy keyframe playback Hz; model remains trained at 30 Hz; "
                             "default 15 Hz executes all targets at half nominal speed")
    parser.add_argument("--servo-hz", type=float, default=60.0,
                        help="smooth Cartesian dispatch Hz; must be an integer multiple "
                             "of --control-hz (default: 60)")
    parser.add_argument("--prefix-length", type=int, default=6, choices=range(1,7))
    parser.add_argument("--async-latency-margin-s", type=float, default=.05,
                        help="required inference headroom inside the action-prefix window "
                             "before asynchronous motion starts (default: 0.05 s)")
    parser.add_argument("--async-startup-retries", type=int, default=1,
                        help="number of over-budget cold-start results to discard and retry "
                             "while the robot remains stationary (default: 1)")
    parser.add_argument("--max-async-handover-late-s", type=float, default=.05,
                        help="maximum validated async result lateness at a chunk handover "
                             "(default: 0.05 s)")
    parser.add_argument("--async-trajectory-retries", type=int, default=1,
                        help="stationary fresh-observation retries after a retryable "
                             "async trajectory rejection (default: 1)")
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--max-observation-age", type=float, default=.15)
    parser.add_argument("--feedback-poll-hz", type=float, default=60,
                        help="independent HAL state polling rate (default: 60 Hz)")
    parser.add_argument("--max-feedback-age", type=float, default=.15,
                        help="feedback stale threshold (default: 0.15 s); one additional "
                             "feedback-poll window is allowed before soft-stop")
    parser.add_argument("--max-observation-drift-m", type=float, default=.01)
    parser.add_argument("--max-observation-drift-rad", type=float, default=.15)
    parser.add_argument("--max-follow-error-m", type=float, default=.04,
                        help="soft-stop when measured position differs from the previous target "
                             "by more than this value (default: 0.04 m)")
    parser.add_argument("--max-follow-error-rad", type=float, default=math.radians(15),
                        help="soft-stop when measured orientation differs from the previous target "
                             "by more than this value (default: 15 deg)")
    parser.add_argument("--dispatch-rebase-s", type=float, default=.02,
                        help="rebase cadence after this bounded scheduling delay")
    parser.add_argument("--max-dispatch-late-s", type=float, default=.15,
                        help="dispatch/command RPC delay threshold (default: 0.15 s); "
                             "a single command RPC overrun is tolerated, two consecutive "
                             "overruns or one over 2x this value soft-stop")
    parser.add_argument("--gripper-deadband-m", type=float, default=.0005)
    parser.add_argument("--settle-timeout", type=float, default=5)
    parser.add_argument("--settle-position-m", type=float, default=.01)
    parser.add_argument("--settle-rotation-rad", type=float, default=.1)
    parser.add_argument("--settle-gripper-m", type=float, default=.01,
                        help="final left-gripper error allowed for 0.2 s (default: 0.01 m)")
    parser.add_argument("--settle-linear-speed-m-s", type=float, default=.005,
                        help="maximum measured end-effector linear speed while settled "
                             "(default: 0.005 m/s)")
    parser.add_argument("--settle-angular-speed-rad-s", type=float,
                        default=math.radians(1),
                        help="maximum measured end-effector angular speed while settled "
                             "(default: 1 deg/s)")
    parser.add_argument("--settle-joint-speed-deg-s", type=float, default=3,
                        help="maximum measured speed of any arm joint while settled "
                             "(default: 3 deg/s)")
    parser.add_argument("--settle-stable-s", type=float, default=.2,
                        help="continuous fresh-feedback stability duration (default: 0.2 s)")
    parser.add_argument("--home-velocity", type=float, default=.2)
    parser.add_argument("--reset-timeout", type=float, default=10)
    parser.add_argument("--reset-joint-tolerance-deg", type=float, default=2)
    parser.add_argument("--reset-gripper-tolerance-m", type=float, default=.003)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--probe", action="store_true")
    group.add_argument("--observe-infer-test", action="store_true",
                       help="run one inference from live observations without acquiring control or moving the robot")
    group.add_argument("--live-infer-test", action="store_true")
    args = parser.parse_args(argv)
    if args.chunks < 0:
        parser.error("--chunks must be >= 0")
    if args.async_startup_retries < 0:
        parser.error("--async-startup-retries must be >= 0")
    if args.async_trajectory_retries < 0:
        parser.error("--async-trajectory-retries must be >= 0")
    for name, value in vars(args).items():
        if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_','-')} must be finite and positive")
    if args.dispatch_rebase_s >= args.max_dispatch_late_s:
        parser.error("--dispatch-rebase-s must be less than --max-dispatch-late-s")
    ratio = args.servo_hz / args.control_hz
    if (args.servo_hz > 120 or int(round(ratio)) < 1 or
            not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9)):
        parser.error("--servo-hz must be <= 120 and an integer multiple of --control-hz")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        print(f"{args.control_hz:g}Hz policy keyframes -> C2 quintic -> "
              f"{args.servo_hz:g}Hz Cartesian ONLINE_TRACKING; model=30Hz; "
              f"inference={args.inference_mode}; time_scale=1; "
              f"mandatory dual-arm + dual-gripper reset; async prefix={args.prefix_length} "
              f"({args.prefix_length/args.control_hz:.3f}s), "
              f"latency_margin={args.async_latency_margin_s:.3f}s, "
              f"startup_retries={args.async_startup_retries}, "
              f"trajectory_retries={args.async_trajectory_retries}, "
              f"handover_late_limit={args.max_async_handover_late_s:.3f}s; "
              f"HAL_RPC={'single-inflight-pipelined' if args.inference_mode == 'async' else 'synchronous'}; "
              "no HAL/model connection")
        return
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    LOG.info("execution: policy=%.3fHz servo=%.3fHz model=30Hz mode=%s "
             "chunk=%.3fs prefix_window=%.3fs async_margin=%.3fs "
             "async_handover_late=%.3fs trajectory_retries=%d "
             "Cartesian C2 time_scale=1",
             args.control_hz, args.servo_hz, args.inference_mode, 30/args.control_hz,
             args.prefix_length/args.control_hz, args.async_latency_margin_s,
             args.max_async_handover_late_s, args.async_trajectory_retries)
    observations = control = policy = predictor = runner = None
    try:
        observations = ObservationSource(args.hal_host, args.hal_port, args.max_observation_age)
        observations.start()
        control = CartesianHAL(args.hal_host, args.hal_port,
                               args.feedback_poll_hz, args.max_feedback_age)
        control.connect()
        state, stamp, sequence = observations.latest()
        raw = build_raw(state, args.task)
        print("HAL Cartesian capability:", {"linear_m_s":control.cap_linear,
                                             "angular_rad_s":control.cap_angular,
                                             "pose":"World xyz + XYZW"})
        print("observation:", {"age_s":time.monotonic()-stamp, "sequence":sequence,
                               "images":{k:list(v.shape) for k,v in raw["images"].items()},
                               "left_pose":raw["state"]["left_cart"].tolist()})
        if args.probe:
            return
        policy = PolicyClient(args.server_host, args.server_port, args.request_timeout)
        if args.observe_infer_test:
            started = time.monotonic()
            chunk, metadata = policy.request(raw, 1)
            print("model:", {
                "latency_s": time.monotonic()-started,
                "server_latency_s": metadata.get("latency_s"),
                "shape": list(chunk["left_pose"].shape),
                "first_pose": chunk["left_pose"][0].tolist(),
                "policy_commands_sent": False,
                "control_acquired": False,
            })
            return
        if args.inference_mode == "async" and not policy.metadata.get("model_warmed", False):
            # The official process is resident when the adapter says ready, but
            # CUDA kernels and image preprocessing are lazy on the first real
            # observation. Warm them before acquiring robot control; this
            # prediction is never dispatched.
            warm_started = time.monotonic()
            _, warm_metadata = policy.request(raw, 0)
            warm_latency = time.monotonic()-warm_started
            LOG.info(
                "stationary model warm-up complete before control acquisition: "
                "end_to_end=%.3fs model=%s server=%s",
                warm_latency,
                _format_optional(warm_metadata.get("model_latency_s")),
                _format_optional(warm_metadata.get("latency_s")))
        predictor = Predictor(policy)
        policy = None  # Predictor owns and closes the policy connection.
        runner = Runner(args, observations, control, predictor)
        control.acquire()
        control.reset_for_inference(
            args.home_velocity, args.reset_timeout,
            args.reset_joint_tolerance_deg, args.reset_gripper_tolerance_m)
        reset_finished = time.monotonic()
        observations.latest(min_stamp=reset_finished)
        # Capture the model observation only after the verified reset.  The
        # observation stream stays live during reset, so latest() returns the
        # new aligned images and state rather than a pre-reset sample.
        if args.live_infer_test:
            result = runner.initial()
            _, chunk, metadata, latency = result
            print("model:", {"latency_s":latency, "shape":list(chunk["left_pose"].shape),
                             "first_pose":chunk["left_pose"][0].tolist(),
                             "policy_commands_sent":False,
                             "reset_completed":True})
            return
        first = runner.initial_for_execution()  # bounded warm-up while robot is stationary
        runner.execute(first)
    except KeyboardInterrupt:
        LOG.info("interrupted; soft-stopping Cartesian tracking")
    except Exception as error:
        if runner is not None:
            runner.trace.append({"event": "error", "time": time.monotonic(),
                                 "type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if runner is not None:
            runner.close()
            runner.save_trace()
        if control is not None:
            control.close()
        if predictor is not None:
            predictor.close()
        elif policy is not None:
            policy.close()
        if observations is not None:
            observations.close()


if __name__ == "__main__":
    main()
