"""XR1 relative Cartesian actions -> absolute World-frame Cartesian targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


def finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite {shape}, got {array.shape}")
    return array


def robot_state(raw: Mapping) -> dict[str, np.ndarray]:
    """Convert HAL degrees/metres/XYZW into the official XR1 client contract."""
    state = raw["state"]
    result = {}
    for side in ("left", "right"):
        joints = finite(state[f"{side}_arm"], (7,), f"{side} joints")
        gripper = finite(state[f"{side}_gripper"], (1,), f"{side} gripper")
        pose = finite(state[f"{side}_cart"], (7,), f"{side} Cartesian pose")
        if not -.001 <= gripper[0] <= .091:
            raise ValueError(f"{side} gripper is outside metre range")
        if abs(np.linalg.norm(pose[3:]) - 1) > 1e-3:
            raise ValueError(f"{side} XYZW quaternion is not unit length")
        result[f"{side}_arm_joint"] = np.deg2rad(joints).astype(np.float32)
        result[f"{side}_gripper_pos"] = gripper.astype(np.float32)
        result[f"{side}_ee_pos"] = pose[:3].astype(np.float32)
        result[f"{side}_ee_rotm"] = Rotation.from_quat(pose[3:]).as_matrix().astype(np.float32)
    return result


def recover_left(action, state):
    """Recover 30 absolute targets; every delta is anchored to one observation."""
    action = finite(action, (30, 60), "XR1 action")
    observed_rotation = finite(state["left_ee_rotm"], (3, 3), "left rotation")
    observed_position = finite(state["left_ee_pos"], (3,), "left position")
    observed_gripper = finite(state["left_gripper_pos"], (1,), "left gripper")[0]
    positions = observed_position + action[:, :3] @ observed_rotation.T
    rotations = observed_rotation @ Rotation.from_rotvec(action[:, 3:6]).as_matrix()
    gripper = observed_gripper + action[:, 6]
    # The policy predicts an unconstrained relative gripper delta, so small or
    # large endpoint overshoots are expected once the fingers are fully closed
    # or open.  A gripper command is a bounded actuator target: saturating it is
    # safe and preserves continuous inference, while every non-finite output is
    # still rejected by finite() above.
    return positions, rotations, np.clip(gripper, 0, .09)


def rotation_matrices_to_quaternions(rotations):
    quaternions = Rotation.from_matrix(np.asarray(rotations)).as_quat()  # XYZW
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    return quaternions


def quaternion_angle(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    dot = np.sum(left * right, axis=-1)
    return 2 * np.arccos(np.clip(np.abs(dot), 0, 1))


@dataclass(frozen=True)
class CartesianLimits:
    max_chunk_translation_m: float = .25
    max_chunk_rotation_rad: float = np.deg2rad(60)


class CartesianBoundaryError(ValueError):
    """Structured, retryable rejection of an unsafe Cartesian prediction."""
    retryable = True

    def __init__(self, kind, frame_index, value, limit, observed, target):
        self.kind = str(kind)
        self.frame_index = int(frame_index)
        self.value = float(value)
        self.limit = float(limit)
        self.observed = np.asarray(observed, dtype=np.float64).tolist()
        self.target = np.asarray(target, dtype=np.float64).tolist()
        if self.kind == "translation":
            self.error_code = "cartesian_translation_boundary"
            value_text = f"distance={self.value:.6f}m limit={self.limit:.6f}m"
        elif self.kind == "rotation":
            self.error_code = "cartesian_rotation_boundary"
            value_text = (f"angle={np.rad2deg(self.value):.3f}deg "
                          f"limit={np.rad2deg(self.limit):.3f}deg")
        else:  # pragma: no cover - only the two checked dimensions are supported.
            raise ValueError(f"unknown Cartesian boundary kind {self.kind!r}")
        super().__init__(
            f"Cartesian {self.kind} exceeds chunk boundary: "
            f"frame_index={self.frame_index} frame_number={self.frame_index + 1} "
            f"{value_text} observed={self.observed} target={self.target}")

    def metadata(self):
        return {
            "error_code": self.error_code,
            "retryable": self.retryable,
            "boundary_kind": self.kind,
            "frame_index": self.frame_index,
            "frame_number": self.frame_index + 1,
            "boundary_value": self.value,
            "boundary_limit": self.limit,
            "observed": self.observed,
            "target": self.target,
        }


def make_cartesian_chunk(raw, action, limits=CartesianLimits(), official_targets=None):
    """Create checked World targets, preferring the official client's recovery."""
    state = robot_state(raw)
    if official_targets is None:
        positions, rotations, gripper = recover_left(action, state)
    else:
        positions = finite(official_targets["left_ee_pos"], (30, 3), "official left positions")
        rotations = finite(official_targets["left_ee_rotm"], (30, 3, 3), "official left rotations")
        gripper = finite(official_targets["left_gripper_pos"], (30, 1),
                         "official left gripper")[:, 0]
        gripper = np.clip(gripper, 0, .09)
    quaternions = rotation_matrices_to_quaternions(rotations)
    poses = np.concatenate([positions, quaternions], axis=1)
    observed = finite(raw["state"]["left_cart"], (7,), "observed left pose")

    translation_from_observation = np.linalg.norm(positions - observed[:3], axis=1)
    rotation_from_observation = quaternion_angle(quaternions, observed[3:])
    translation_frame = int(np.argmax(translation_from_observation))
    if translation_from_observation[translation_frame] > limits.max_chunk_translation_m:
        raise CartesianBoundaryError(
            "translation", translation_frame,
            translation_from_observation[translation_frame],
            limits.max_chunk_translation_m, observed[:3], positions[translation_frame])
    rotation_frame = int(np.argmax(rotation_from_observation))
    if rotation_from_observation[rotation_frame] > limits.max_chunk_rotation_rad:
        raise CartesianBoundaryError(
            "rotation", rotation_frame, rotation_from_observation[rotation_frame],
            limits.max_chunk_rotation_rad, observed[3:], quaternions[rotation_frame])

    return {
        "left_pose": poses.astype(np.float32),
        "left_gripper": np.asarray(gripper, dtype=np.float32),
        "observation_pose": observed.astype(np.float32),
        "observation_gripper": np.asarray(raw["state"]["left_gripper"], dtype=np.float32),
        "raw_action": np.asarray(action, dtype=np.float32),
    }
