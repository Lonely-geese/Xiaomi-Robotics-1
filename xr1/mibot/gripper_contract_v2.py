"""Training/inference contract for uncertain USB gripper feedback (metres).

Use anchor_proprio BEFORE q01/q99 normalization. Use absolute_gripper AFTER
normalization is undone; do not add the hardware's measured opening.
"""
import numpy as np

ANCHOR_M = .09

def anchor_proprio(proprio):
    value=proprio.clone() if hasattr(proprio,'detach') else np.array(proprio,copy=True)
    assert value.shape[-1]==60
    value[...,7]=ANCHOR_M
    return value

def absolute_gripper(denormalized_action,closed_width_m=.00104):
    value=denormalized_action if hasattr(denormalized_action,'detach') else np.asarray(denormalized_action)
    assert value.shape[-1] in (7,60)
    if hasattr(value,'detach'):
        return (value[...,6]+ANCHOR_M).clamp(closed_width_m,ANCHOR_M)
    return np.clip(value[...,6]+ANCHOR_M,closed_width_m,ANCHOR_M)
