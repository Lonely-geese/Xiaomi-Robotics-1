"""Human-gated dataset aggregation, without a critic or advantage labels."""
import hashlib
import json

import numpy as np


VERSION = 'hg_dagger_v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def is_validation(dataset, episode_index):
    key = f'{dataset}:{episode_index}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0


def classify_segments(pool, outcome, human_intervened, human):
    """Partition every frame; terminal labels are audit metadata, not RL rewards.

    Only demonstrations and confirmed-success human actions enter BC. A human
    takeover terminates the preceding autonomous segment as a failure, even
    when the complete episode eventually succeeds.
    """
    human = np.asarray(human)
    if human.ndim != 1 or not len(human) or not np.isin(human, [-1, 0, 1]).all():
        raise ValueError('Invalid action-aligned intervention mask')
    n = len(human)

    def span(a, b, event, target=None, terminal=None):
        return dict(start=int(a), end_exclusive=int(b), event=event,
                    target_pool=target, terminal_label=terminal)

    if pool == 'sft':
        return [span(0, n, 'demo_success', 'demo', 0)] if outcome == 'success' else [span(0, n, 'unconfirmed_demo')]
    if pool != 'rollout':
        raise ValueError(f'Unknown pool: {pool}')
    if outcome not in ('success', 'failure'):
        return [span(0, n, 'unknown_outcome')]
    if human_intervened is None or np.any(human < 0):
        return [span(0, n, 'unknown_intervention')]
    if bool(human_intervened) != bool(np.any(human == 1)):
        return [span(0, n, 'intervention_outside_retained_actions')]
    edges = [0, *(np.flatnonzero(human[1:] != human[:-1]) + 1).tolist(), n]
    result = []
    for a, b in zip(edges[:-1], edges[1:]):
        if human[a] == 1:
            result.append(span(a, b, 'correction_success', 'correction', 0)
                          if outcome == 'success' else span(a, b, 'unconfirmed_correction'))
        elif b < n:
            result.append(span(a, b, 'takeover_failure', terminal=-1))
        else:
            result.append(span(a, b, 'autonomous_success' if outcome == 'success' else 'failure',
                               terminal=0 if outcome == 'success' else -1))
    return result


def expert_chunk_starts(human, horizon):
    """RLinf require_all_intervene: every non-padding action must be human.

    Episode-end padding is permitted; handback to the policy is not padding.
    Prefix sums avoid allocating an episode-length x horizon mask.
    """
    flags = np.asarray(human) == 1
    if flags.ndim != 1 or horizon <= 0:
        raise ValueError('Expected 1-D flags and a positive action horizon')
    bad = np.r_[0, np.cumsum(~flags)]
    starts = np.arange(len(flags))
    return starts[bad[np.minimum(starts + horizon, len(flags))] == bad[starts]]
