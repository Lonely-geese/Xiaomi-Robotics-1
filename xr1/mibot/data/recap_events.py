"""Task-specific RECAP event returns; independent of the RLinf model topology."""
import hashlib
import json

import numpy as np


REWARD_VERSION = 'recap_events_v1'
TERMINAL_REWARDS = dict(demo_success=0.0, autonomous_success=0.0,
                        takeover=-0.5, failure=-1.0)


def validation_episode(record):
    # Segments from one original episode must never cross the split.
    if 'split' in record:
        if record['split'] not in ('train', 'validation'):
            raise ValueError('Unknown data split')
        return record['split'] == 'validation'
    key = f"{record['dataset']}:{record['episode_index']}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0


def event_segments(pool, outcome, human_intervened, human):
    """Return (retained spans, excluded spans), with explicit exclusion reasons.

    A span is [start, end, event]. An observed 0->1 transition is a takeover;
    neither the following correction nor its terminal success can bootstrap it.
    Human segments of successful rollouts are policy-only correction examples.
    Unknown outcomes/coverage and corrections without confirmed success are out.
    """
    human = np.asarray(human)
    n = len(human)
    if n == 0 or not np.isin(human, [-1, 0, 1]).all():
        raise ValueError('Invalid intervention mask')
    if pool == 'sft':
        if outcome == 'success':
            return [(0, n, 'demo_success')], []
        return [], [(0, n, 'unconfirmed_demo')]
    if pool != 'rollout':
        raise ValueError(f'Unknown pool: {pool}')
    if outcome not in ('success', 'failure'):
        return [], [(0, n, 'unknown_outcome')]
    if human_intervened is None or np.any(human < 0):
        return [], [(0, n, 'unknown_intervention')]
    if bool(human_intervened) != bool(np.any(human == 1)):
        return [], [(0, n, 'intervention_outside_retained_actions')]
    edges = [0, *(np.flatnonzero(human[1:] != human[:-1]) + 1).tolist(), n]
    kept, excluded = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if human[a] == 1:
            if outcome == 'success':
                kept.append((a, b, 'correction'))
            else:
                excluded.append((a, b, 'correction_without_confirmed_success'))
        else:
            event = 'takeover' if b < n else 'autonomous_success' if outcome == 'success' else 'failure'
            kept.append((a, b, event))
    return kept, excluded


def event_returns(length, event, time_reference_frames):
    """Raw rewards in [-1.1, 0]; terminal events dominate total time cost.

    The reference is a documented dataset frame bound, not a fabricated robot
    timeout. gamma=1, at most 0.1 time cost per retained segment. Dividing by
    1.1 fits the unchanged native categorical head's [-1, 0] support.
    """
    if event not in TERMINAL_REWARDS or not 0 < length <= time_reference_frames:
        raise ValueError('Invalid event or time-reference bound')
    rewards = np.full(length, -0.1 / time_reference_frames, dtype=np.float32)
    rewards[-1] = TERMINAL_REWARDS[event]
    returns = np.cumsum(rewards[::-1], dtype=np.float64)[::-1].astype(np.float32)
    return rewards, returns


def event_positive_labels(event, advantage, threshold):
    advantage = np.asarray(advantage)
    if event in ('demo_success', 'correction'):
        return np.ones(advantage.shape, dtype=bool)
    if event not in TERMINAL_REWARDS:
        raise ValueError(f'Unknown event: {event}')
    return np.isfinite(advantage) & (advantage > 0) & (advantage >= threshold)


def require_matching_data(checkpoint_metadata, metadata):
    if checkpoint_metadata['source_metadata_sha256'] != metadata['source_metadata_sha256']:
        raise ValueError('Checkpoint source data mismatch')
    if checkpoint_metadata.get('experiment_fingerprint') != metadata.get('experiment_fingerprint'):
        raise ValueError('Checkpoint reward/selection fingerprint mismatch; retrain the critic')


def experiment_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
