# Copyright (C) 2026 Xiaomi Corporation.
"""
RoboCasa evaluation entry point using deploy/client.py (socket-based).
"""
import os
import sys
import time
import json
import logging
import tyro
import imageio
import numpy as np
from dataclasses import dataclass
from PIL import Image
from scipy.spatial.transform import Rotation as R

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
from client import Client

ROBOT_TYPE = "robocasa_mg"
STATE_DIM = 60
ACTION_DIM = 7

EEF_AXES_XFORM = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}

camera_names = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def create_env(
    env_name,
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=256,
    camera_heights=256,
    seed=None,
    render_onscreen=False,
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    env = robosuite.make(**env_kwargs)
    return env


def render_obs(env, camera_names, base2world, camera_height=256, camera_width=256):
    rgbs = []
    for cam_name in camera_names:
        rgb = env.sim.render(
            height=camera_height, width=camera_width, camera_name=cam_name, depth=False, segmentation=False
        )
        rgb = rgb[::-1].copy()
        rgbs.append(rgb)
    rgbs = np.stack(rgbs, axis=0)

    robot = env.robots[0]
    proprios_direct = robot.get_robot_joint_positions()
    proprios_gripper = robot.get_gripper_joint_positions("right")[0:1]
    proprios = np.concatenate([proprios_direct, proprios_gripper], axis=0)

    controller = env.robots[0].composite_controller
    eef_pos, eef_mat = (
        controller.part_controllers[controller.arms[0]].ref_pos,
        controller.part_controllers[controller.arms[0]].ref_ori_mat,
    )
    eef2world = np.eye(4)
    eef2world[:3, :3] = eef_mat
    eef2world[:3, 3] = eef_pos

    eef2base = np.linalg.inv(base2world) @ eef2world
    eef2base_pos = eef2base[:3, 3]
    rot_mat = eef2base[:3, :3] @ EEF_AXES_XFORM
    rot_quat = R.from_matrix(rot_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]
    gripper_openness = (
        controller.part_controllers[list(controller.grippers.keys())[0]].joint_pos[0:1]
        / controller.part_controllers[list(controller.grippers.keys())[0]].actuator_max[0]
    )
    eef_states = np.concatenate([eef2base_pos, rot_quat, gripper_openness])

    return rgbs, proprios, eef_states


def rgbs_to_pil_images(rgbs):
    return [Image.fromarray(rgbs[i]) for i in range(rgbs.shape[0])]


def center_crop_pil(images, crop_ratio):
    if crop_ratio >= 1.0:
        return images
    cropped = []
    for img in images:
        w, h = img.size
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)
        cropped.append(img)
    return cropped


class EvalClient:
    """Wraps deploy/client.py with processor-based input construction."""

    def __init__(self, host="localhost", port=10086, model_path=""):
        self.client = Client(host=host, port=port, model_path=model_path)
        self.processor = self.client.processor
        logging.info(f"EvalClient ready. Robot types: {self.processor.list_robot_types()}")

    def __call__(self, state, images, instruction, global_rank=0, rollout_i=0, step_i=0):
        """
        Args:
            state: numpy array shape (8,) — [7 joint positions, 1 gripper pos]
            images: list of 3 PIL.Image (256x256) — [base1, base2, wrist_left]
            instruction: str — task description
        Returns:
            numpy array shape (10, 7)
        """
        state_padded = np.zeros(STATE_DIM, dtype=np.float32)
        state_padded[: len(state)] = state

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following observations are captured from multiple views.\n# Base View\n"},
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

        data = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state_padded.reshape(1, 1, STATE_DIM),
            robot_type=ROBOT_TYPE,
        )

        request_data = dict(data)
        request_data["task_id"] = ROBOT_TYPE
        request_data["seed"] = 42

        response = self.client(**request_data)
        action = response[0, :, :ACTION_DIM].float().numpy()
        return action

    def close(self):
        self.client.close()


@dataclass
class Args:
    task_name: str
    model_path: str
    num_eval_episodes: int = 5
    start_seed: int = 0
    action_length: int = 10
    save_root_dir: str = "./eval_results/robocasa/"
    server_addr: str = "localhost"
    server_port: int = 10086
    crop_ratio: float = 0.95


def main(args: Args):
    env_name = args.task_name
    if env_name not in TASK_MAX_STEPS:
        logging.error(f"Unknown task: {env_name}. Available: {list(TASK_MAX_STEPS.keys())}")
        sys.exit(1)

    logging.info(f"Task: {env_name} | Episodes: {args.num_eval_episodes} | Start seed: {args.start_seed}")
    logging.info(f"Server: {args.server_addr}:{args.server_port}")

    client = EvalClient(host=args.server_addr, port=args.server_port, model_path=args.model_path)

    info = {}
    num_success_rollouts = 0
    for rollout_i in range(args.num_eval_episodes):
        seed = args.start_seed + rollout_i
        env = create_env(
            env_name=env_name,
            render_onscreen=False,
            seed=seed,
        )
        env.reset()

        controller = env.robots[0].composite_controller
        base_pos, base_mat = (
            controller.part_controllers[controller.arms[0]].origin_pos,
            controller.part_controllers[controller.arms[0]].origin_ori,
        )
        base2world = np.eye(4)
        base2world[:3, :3] = base_mat
        base2world[:3, 3] = base_pos

        num_steps = TASK_MAX_STEPS[env_name]
        instruction = env.get_ep_meta()["lang"]
        logging.info(f"Task {env_name}: Episode {rollout_i+1} (seed={seed}): Instruction: {instruction}")

        step_i = 0
        success = False
        video_array = []

        while step_i < num_steps:
            rgbs, proprios, eef_states = render_obs(env, camera_names, base2world)

            images = rgbs_to_pil_images(rgbs)
            images = center_crop_pil(images, args.crop_ratio)
            action = client(
                proprios, images, instruction, global_rank=0, rollout_i=rollout_i, step_i=step_i
            )

            action = action[: args.action_length]
            pad_action = np.zeros(env.action_spec[0].shape)
            for ai in range(action.shape[0]):
                pad_action[:7] = action[ai]

                if step_i % 5 == 0:
                    rgbs, _, _ = render_obs(env, camera_names, base2world)
                    video_array.append(rgbs.transpose(1, 0, 2, 3).reshape(256, -1, 3))

                env.step(pad_action)
                step_i += 1

                if env._check_success():
                    success = True
                    num_success_rollouts += 1
                    break

                if step_i >= num_steps:
                    break

            if success or step_i >= num_steps:
                break

        env.close()

        os.makedirs(f"{args.save_root_dir}/{env_name}", exist_ok=True)
        video_path = (
            f"{args.save_root_dir}/{env_name}/seed{seed}_{'success' if success else 'failure'}.mp4"
        )
        imageio.mimsave(video_path, video_array, fps=10)

        ep_num = rollout_i + 1
        current_rate = num_success_rollouts / ep_num * 100
        logging.info(
            f"Task {env_name}: Episode {ep_num} (seed={seed}): "
            f"{'Success' if success else 'Failure'} | "
            f"Current Success Rate: {current_rate:.2f}% ({num_success_rollouts}/{ep_num})"
        )

    client.close()

    success_rate = num_success_rollouts / args.num_eval_episodes
    info[env_name] = {
        "num_success_rollouts": num_success_rollouts,
        "num_rollouts": args.num_eval_episodes,
        "success_rate": success_rate,
    }

    logging.info(
        f"Task {env_name} Final Success Rate: {success_rate*100:.2f}% "
        f"({num_success_rollouts}/{args.num_eval_episodes})"
    )

    os.makedirs(args.save_root_dir, exist_ok=True)
    result_path = os.path.join(args.save_root_dir, f"eval_results_{env_name}.json")
    with open(result_path, "w") as f:
        json.dump(info, f, indent=4)
    logging.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
