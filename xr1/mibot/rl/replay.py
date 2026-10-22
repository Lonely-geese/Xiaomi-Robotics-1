"""Explicit transition contracts and frame-discounted n-chunk returns."""

import torch


def chunk_return(rewards, terminated, start, horizon, n_chunks, gamma):
    """Input arrays contain one reward/terminal flag per executed primitive action.

    Returns discounted reward, number of executed frames, terminal flag and next
    observation index. The caller must supply a real next state for bootstrapping.
    """
    if horizon < 1 or n_chunks < 1 or not 0 < gamma <= 1:
        raise ValueError("invalid horizon, n_chunks or gamma")
    if len(rewards) != len(terminated) or not 0 <= start < len(rewards):
        raise ValueError("invalid reward/termination sequence")
    result, count, done = 0., 0, False
    for index in range(start, min(len(rewards), start + horizon * n_chunks)):
        result += gamma ** count * float(rewards[index])
        count += 1
        done = bool(terminated[index])
        if done:
            break
    return result, count, done, start + count


def validate_batch(batch):
    required = {"actor_observation", "critic_observation", "next_critic_observation",
                "action", "action_mask", "reward", "steps", "terminated"}
    if required - batch.keys():
        raise ValueError(f"missing transition fields: {sorted(required - batch.keys())}")
    action, mask = batch["action"], batch["action_mask"]
    if action.ndim != 3 or mask.shape != action.shape:
        raise ValueError("action and action_mask must be [B,H,A]")
    if not torch.isfinite(action).all() or not torch.isfinite(mask).all():
        raise ValueError("non-finite action or mask")
    if not ((mask == 0) | (mask == 1)).all() or not mask.bool().flatten(1).any(-1).all():
        raise ValueError("action_mask must be binary with valid actions in every sample")
    for key in ("reward", "steps", "terminated"):
        value = batch[key]
        if value.shape not in ((len(action),), (len(action), 1)) or not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain one finite scalar per transition")
    if not ((batch["terminated"] == 0) | (batch["terminated"] == 1)).all():
        raise ValueError("terminated must be binary; time limits must bootstrap")
    if (batch["steps"] < 1).any() or (batch["steps"] != batch["steps"].round()).any():
        raise ValueError("steps must count executed primitive actions")


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value
