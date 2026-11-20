"""Experimental joint actor/correction inference, distinct from online BEE.

Both heads must receive the SAME original VLA proposal. No targets, episode
labels, or future valid-horizon masks enter these prediction-only operations.
"""
import math
import torch


def combine_actions(actor, corrected, reference, editable, *, actor_weight=.5):
    if not math.isfinite(actor_weight) or not 0 <= actor_weight <= 1:
        raise ValueError('Actor weight must be in [0, 1]')
    if actor.shape != corrected.shape or actor.shape != reference.shape:
        raise ValueError('Prediction shapes differ')
    result=actor_weight*actor+(1-actor_weight)*corrected
    return torch.where(editable, result, reference)


def constrain_to_correction(actor, corrected, variance, reference, editable, *, epsilon=.3):
    """Radial contraction into a Mahalanobis ellipsoid, NOT Euclidean projection.

    Uses all K predicted steps, including unobserved future steps during offline
    evaluation. Variance is the correction model's estimate, not actor variance.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    if actor.shape != corrected.shape or variance.shape != actor.shape or reference.shape != actor.shape:
        raise ValueError('Prediction shapes differ')
    if not torch.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError('Variance must be finite and positive')
    delta=(actor-corrected).masked_fill(~editable,0)
    rho=(delta.square()/variance).flatten(1).mean(1)
    scale=(epsilon/rho.clamp_min(1e-12)).sqrt().clamp(max=1)
    result=corrected+scale[:,None,None]*delta
    return dict(action=torch.where(editable,result,reference),rho=rho,scale=scale)
