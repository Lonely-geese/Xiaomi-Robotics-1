# Copyright (C) 2026 Xiaomi Corporation.
"""VLABench evaluation entry point using the shared deploy/client.py protocol."""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import tyro
from PIL import Image

from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.base import Policy
from VLABench.robots import *  # noqa: F401,F403
from VLABench.tasks import *  # noqa: F401,F403
from VLABench.utils.utils import quaternion_to_euler

_DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy"
sys.path.insert(0, str(_DEPLOY_DIR))
from client import Client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ROBOT_TYPE = "vlabench_choice"
DEFAULT_STATE_DIM = 60
DEFAULT_ACTION_DIM = 7
DEFAULT_VIEW_NAMES = ("Ego View", "Base View", "Left-Wrist View")


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"Expected a boolean or 0/1 value, got {value!r}")


def _load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def _find_track_config(eval_track: str) -> str:
    vlabench_root = os.environ.get("VLABENCH_ROOT")
    if not vlabench_root:
        raise EnvironmentError(
            "VLABENCH_ROOT is not set. Export VLABENCH_ROOT=/path/to/VLABench before evaluation."
        )

    candidates = (
        Path(vlabench_root) / "configs" / "evaluation" / "tracks" / f"{eval_track}.json",
        Path(vlabench_root) / "VLABench" / "configs" / "evaluation" / "tracks" / f"{eval_track}.json",
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError("Track configuration not found. Tried:\n" + "\n".join(map(str, candidates)))


def resolve_tasks_and_episode_configs(
    *,
    eval_track: Optional[str],
    episode_config_path: Optional[str],
    tasks_text: Optional[str],
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Resolve task names and optional fixed episode configurations."""
    episode_configs: Optional[Dict[str, Any]] = None

    if episode_config_path:
        episode_configs = _load_json(episode_config_path)
    elif eval_track:
        episode_configs = _load_json(_find_track_config(eval_track))

    requested_tasks = tasks_text.split() if tasks_text else None
    if episode_configs is not None:
        available_tasks = list(episode_configs.keys())
        if requested_tasks is None:
            tasks = available_tasks
        else:
            tasks = [task for task in requested_tasks if task in episode_configs]
            missing = [task for task in requested_tasks if task not in episode_configs]
            if missing:
                logger.warning("Tasks absent from the episode configuration: %s", missing)
            episode_configs = {task: episode_configs[task] for task in tasks}
    else:
        if not requested_tasks:
            raise ValueError("Provide --eval-track, --episode-config-path, or --tasks.")
        tasks = requested_tasks

    return tasks, episode_configs


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _to_pil(image: np.ndarray, image_size: int) -> Image.Image:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected an ndarray image, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC image with 1, 3, or 4 channels, got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        image = np.clip(image, 0.0, 1.0 if max_value <= 1.0 + 1e-6 else 255.0)
        if max_value <= 1.0 + 1e-6:
            image = image * 255.0
    image = image.astype(np.uint8, copy=False)

    if image.shape[-1] == 1:
        pil_image = Image.fromarray(image[..., 0], mode="L").convert("RGB")
    else:
        mode = "RGB" if image.shape[-1] == 3 else "RGBA"
        pil_image = Image.fromarray(image, mode=mode).convert("RGB")

    if image_size > 0 and pil_image.size != (image_size, image_size):
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return pil_image


def _extract_ee_state(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    if "ee_state" not in obs:
        raise KeyError(f"Observation does not contain 'ee_state'; keys={list(obs.keys())}")

    ee_state = np.asarray(obs["ee_state"], dtype=np.float32).reshape(-1)
    if ee_state.size < 8:
        raise ValueError(f"Expected ee_state=[xyz, quaternion, gripper] with at least 8 values, got {ee_state.size}")

    position_world = ee_state[:3]
    quaternion = ee_state[3:7]
    euler_world = np.asarray(quaternion_to_euler(quaternion), dtype=np.float32).reshape(-1)[:3]
    euler_world = _wrap_to_pi(euler_world)
    gripper = float(ee_state[7])
    return position_world, euler_world, gripper


def _model_state(obs: Dict[str, Any], position_offset: np.ndarray) -> np.ndarray:
    position_world, euler_world, gripper = _extract_ee_state(obs)
    position_robot = position_world - position_offset
    return np.concatenate(
        [position_robot, euler_world, np.asarray([gripper], dtype=np.float32)], axis=0
    ).astype(np.float32)


def _execution_state(obs: Dict[str, Any]) -> np.ndarray:
    position_world, euler_world, gripper = _extract_ee_state(obs)
    return np.concatenate(
        [position_world, euler_world, np.asarray([gripper], dtype=np.float32)], axis=0
    ).astype(np.float32)


def _build_messages(images: List[Image.Image], instruction: str, cot: bool) -> List[Dict[str, Any]]:
    prompt_suffix = "/cot" if cot else "/no_cot"
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
        {"type": "image", "image": images[0]},
        {"type": "text", "text": "\n# Base View\n"},
        {"type": "image", "image": images[1]},
        {"type": "text", "text": "\n# Left-Wrist View\n"},
        {"type": "image", "image": images[2]},
        {
            "type": "text",
            "text": f"\nGenerate robot actions for the task:\n{instruction} {prompt_suffix}",
        },
    ]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
    if not cot:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]}
        )
    return messages


@dataclasses.dataclass
class VLABenchPolicyArgs:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 10086
    robot_type: str = DEFAULT_ROBOT_TYPE
    state_dim: int = DEFAULT_STATE_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    action_chunk_size: int = 10
    replan_steps: int = 5
    image_size: int = 480
    cot: bool = False
    position_offset_x: float = 0.0
    position_offset_y: float = -0.4
    position_offset_z: float = 0.78
    gripper_threshold: float = 0.2
    gripper_open_value: float = 0.04
    request_seed: int = 42


class VLABenchPolicy(Policy):
    """Adapts VLABench observations to the shared XR-1 client/server interface."""

    def __init__(self, args: VLABenchPolicyArgs):
        super().__init__(model=None)
        self.args = args
        if args.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if args.replan_steps <= 0 or args.replan_steps > args.action_chunk_size:
            raise ValueError("replan_steps must be in [1, action_chunk_size]")

        self.client = Client(host=args.host, port=args.port, model_path=args.model_path)
        self.processor = self.client.processor
        if self.processor is None:
            raise RuntimeError("The shared Client did not initialize an AutoProcessor.")

        # Execute only replan_steps from every decoded action chunk, then query again.
        # This matches the original VLABench evaluation behavior for both CoT and no-CoT.
        self.plan_horizon = args.replan_steps
        self._plan: Deque[Tuple[np.ndarray, np.ndarray, float]] = collections.deque()
        self.position_offset = np.asarray(
            [args.position_offset_x, args.position_offset_y, args.position_offset_z],
            dtype=np.float32,
        )
        if hasattr(self.processor, "list_robot_types"):
            logger.info("Processor robot types: %s", self.processor.list_robot_types())

    @property
    def name(self) -> str:
        return "xr1_vlabench"

    def close(self) -> None:
        self.client.close()

    def reset(self) -> None:
        self._plan.clear()

    def _build_model_inputs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rgb_observations = obs.get("rgb")
        if rgb_observations is None or len(rgb_observations) < 4:
            raise ValueError("VLABench obs['rgb'] must contain at least four camera images.")

        second_image, _, front_image, wrist_image = rgb_observations
        images = [
            _to_pil(front_image, self.args.image_size),
            _to_pil(second_image, self.args.image_size),
            _to_pil(wrist_image, self.args.image_size),
        ]
        instruction = str(obs["instruction"])
        state = _model_state(obs, self.position_offset)
        state_padded = np.zeros(self.args.state_dim, dtype=np.float32)
        copy_dim = min(state.size, self.args.state_dim)
        state_padded[:copy_dim] = state[:copy_dim]

        messages = _build_messages(images, instruction, self.args.cot)
        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state_padded.reshape(1, 1, self.args.state_dim),
            robot_type=self.args.robot_type,
        )
        request_data = dict(model_inputs)
        request_data["task_id"] = self.args.robot_type
        request_data["seed"] = self.args.request_seed
        return request_data

    def _request_action_chunk(self, obs: Dict[str, Any]) -> np.ndarray:
        response = self.client(**self._build_model_inputs(obs))
        if torch.is_tensor(response):
            response = response.detach().cpu().float().numpy()
        chunk = np.asarray(response, dtype=np.float32)

        if chunk.ndim == 3:
            if chunk.shape[0] != 1:
                raise RuntimeError(f"Expected batch size 1, got action shape {chunk.shape}")
            chunk = chunk[0]
        elif chunk.ndim == 1:
            chunk = chunk[None, :]

        if chunk.ndim != 2:
            raise RuntimeError(f"Expected decoded actions with shape [T, D], got {chunk.shape}")
        if chunk.shape[1] < self.args.action_dim:
            raise RuntimeError(
                f"Decoded action dimension {chunk.shape[1]} is smaller than required {self.args.action_dim}"
            )
        if chunk.shape[0] < self.plan_horizon:
            raise RuntimeError(
                f"Decoded action horizon {chunk.shape[0]} is smaller than plan horizon {self.plan_horizon}"
            )
        return chunk[:, : self.args.action_dim]

    def predict(self, obs: Dict[str, Any], **_: Any):
        if not self._plan:
            action_chunk = self._request_action_chunk(obs)
            current = _execution_state(obs).copy()
            for action in action_chunk[: self.plan_horizon]:
                current[:6] += action[:6]
                current[3:6] = _wrap_to_pi(current[3:6])
                self._plan.append((current[:3].copy(), current[3:6].copy(), float(action[6])))

        position_world, euler_world, gripper_scalar = self._plan.popleft()
        if gripper_scalar >= self.args.gripper_threshold:
            gripper_state = np.full(2, self.args.gripper_open_value, dtype=np.float32)
        else:
            gripper_state = np.zeros(2, dtype=np.float32)
        return position_world, euler_world, gripper_state


@dataclasses.dataclass
class Args:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 10086
    robot_type: str = DEFAULT_ROBOT_TYPE
    eval_track: Optional[str] = None
    episode_config_path: Optional[str] = None
    tasks: Optional[str] = None
    n_episode: int = 50
    metrics: str = "success_rate intention_score progress_score"
    save_dir: str = "./eval_vlabench/eval_logs/track_1_in_distribution"
    visualization: bool = True
    state_dim: int = DEFAULT_STATE_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    action_chunk_size: int = 10
    replan_steps: int = 5
    image_size: int = 480
    cot: bool = False
    position_offset_x: float = 0.0
    position_offset_y: float = -0.4
    position_offset_z: float = 0.78
    gripper_threshold: float = 0.2
    gripper_open_value: float = 0.04
    request_seed: int = 42


def run(args: Args) -> None:
    args.cot = _normalize_bool(args.cot)
    tasks, episode_configs = resolve_tasks_and_episode_configs(
        eval_track=args.eval_track,
        episode_config_path=args.episode_config_path,
        tasks_text=args.tasks,
    )
    if not tasks:
        logger.warning("No tasks were selected; evaluation skipped.")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    policy = VLABenchPolicy(
        VLABenchPolicyArgs(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            robot_type=args.robot_type,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            action_chunk_size=args.action_chunk_size,
            replan_steps=args.replan_steps,
            image_size=args.image_size,
            cot=args.cot,
            position_offset_x=args.position_offset_x,
            position_offset_y=args.position_offset_y,
            position_offset_z=args.position_offset_z,
            gripper_threshold=args.gripper_threshold,
            gripper_open_value=args.gripper_open_value,
            request_seed=args.request_seed,
        )
    )

    logger.info(
        "Starting VLABench evaluation: tasks=%d episodes=%d server=%s:%d robot_type=%s cot=%s",
        len(tasks),
        args.n_episode,
        args.host,
        args.port,
        args.robot_type,
        args.cot,
    )
    try:
        evaluator = Evaluator(
            tasks=tasks,
            n_episodes=args.n_episode,
            episode_config=episode_configs,
            max_substeps=1,
            save_dir=args.save_dir,
            visulization=args.visualization,
            metrics=args.metrics.split(),
        )
        evaluator.evaluate(policy)
    finally:
        policy.close()


if __name__ == "__main__":
    run(tyro.cli(Args))
