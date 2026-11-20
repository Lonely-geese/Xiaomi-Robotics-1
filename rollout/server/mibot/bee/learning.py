"""BEE arXiv:2609.27450v1 equations 1–9; no robot transport implied.

Online updates require complete, verified chunk transitions and executed actions.
The separate offline.py implements the explicitly different pose-only IQL adapter.
"""
from contextlib import contextmanager
import math
import torch


def correction_nll(distribution, human_action, editable):
    """Eq. 6, sum over observed dimensions then mean over samples (no MSE)."""
    error = (human_action - distribution['corrected'])[..., editable[0]]
    variance = distribution['variance'][..., editable[0]]
    if not torch.isfinite(error).all() or not (variance > 0).all():
        raise ValueError('Invalid correction labels or variance')
    return (.5 * (math.log(2 * math.pi) + variance.log() + error.square()/variance)).flatten(1).sum(1).mean()


def mahalanobis(action, corrected, variance, editable):
    # Eq. 3 uses D=K*d, including noneditable dimensions whose residual is zero.
    diff = (action - corrected).masked_fill(~editable, 0)
    return (diff.square()/variance).flatten(1).mean(1)


def actor_objective(q, action, reference, rho, multiplier, eta=1.):
    anchor = (action-reference).square().flatten(1).sum(1)
    weight = multiplier.detach()
    return ((-q + eta*anchor + weight*rho)/(1+weight)).mean()


def dual_objective(multiplier, rho, epsilon=.3):
    return -(multiplier*(rho.detach()-epsilon)).mean()


def bellman_target(rewards, terminated, next_q, gamma=.99):
    """Full K primitive rewards; terminal reward included, no post-terminal reward."""
    discount = gamma ** torch.arange(rewards.shape[1], device=rewards.device)
    return (rewards*discount).sum(1) + gamma**rewards.shape[1] * (~terminated.bool()) * next_q


@contextmanager
def frozen(module):
    original = [p.requires_grad for p in module.parameters()]
    module.requires_grad_(False)
    try:
        yield
    finally:
        for p, enabled in zip(module.parameters(), original):
            p.requires_grad_(enabled)


def adam(parameters, lr):
    return torch.optim.Adam(parameters, lr=lr, betas=(.9, .95), eps=1e-8)


def optimize(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


class BEELearner:
    """Verified-transition update kernel; deployment/collection adapter is separate.

    Batches must contain full K-step rewards, actual executed 7D actions,
    current/next z, proprio, reference, terminated and next_observation_valid.
    Missing nonterminal successors are rejected; terminal successors aren't read.
    """
    def __init__(self, model):
        if model.config.get('algorithm') != 'bee-paper-v1':
            raise ValueError('Explicit paper algorithm configuration required')
        self.model = model.requires_grad_(True)
        self.model.target_critics.load_state_dict(model.critics.state_dict())
        self.model.target_critics.requires_grad_(False)
        self.optimizers = {
            name: adam(getattr(model, name).parameters(), 3e-6 if name == 'multiplier' else 3e-4)
            for name in ('actor', 'critics', 'correction', 'multiplier')}
        self.updates = 0
        self.new_human = 0

    def correction_step(self, batch):
        m = self.model
        dist = m.distribution(batch['z'], batch['proprio'], batch['reference'])
        loss = correction_nll(dist, batch['human_action'], m.editable)
        optimize(loss, self.optimizers['correction'], m.correction.parameters())
        return float(loss.detach())

    def update(self, batch):
        m = self.model
        cfg = m.config
        if batch['rewards'].shape[1] != m.chunk_len:
            raise ValueError('Full fixed-length chunk required; do not pad partial transitions')
        live = ~batch['terminated'].bool()
        if not batch['next_observation_valid'][live].all():
            raise ValueError('Missing true successor observation; never substitute the next row')
        if not batch['executed_action_valid'].all():
            raise ValueError('Missing executed action dimensions, including gripper')
        with torch.no_grad():
            next_q = torch.zeros(len(live), device=live.device)
            if live.any():
                next_action = m.policy(batch['next_z'][live], batch['next_proprio'][live], batch['next_reference'][live])
                next_q[live] = m.q_values(batch['next_z'][live], batch['next_proprio'][live], next_action, target=True).min(-1).values
            y = bellman_target(batch['rewards'], batch['terminated'], next_q, cfg.get('gamma', .99))
        q = m.q_values(batch['z'], batch['proprio'], batch['executed_action'])
        critic_loss = (q-y[:, None]).square().mean()
        optimize(critic_loss, self.optimizers['critics'], m.critics.parameters())
        with torch.no_grad():
            for target, current in zip(m.target_critics.parameters(), m.critics.parameters()):
                target.lerp_(current, cfg.get('tau', .005))
        self.updates += 1
        result = {'critic_loss': float(critic_loss.detach())}
        if self.updates % cfg.get('policy_delay', 2) == 0:
            with torch.no_grad():
                dist = m.distribution(batch['z'], batch['proprio'], batch['reference'])
                weight = m.lagrange(batch['z'], batch['proprio'])
            with frozen(m.critics):
                action = m.policy(batch['z'], batch['proprio'], batch['reference'], cfg.get('reference_dropout', .5))
                rho = mahalanobis(action, dist['corrected'], dist['variance'], m.editable)
                value = m.q_values(batch['z'], batch['proprio'], action).min(-1).values
                actor_loss = actor_objective(value, action, batch['reference'], rho, weight, cfg.get('eta', 1.))
                optimize(actor_loss, self.optimizers['actor'], m.actor.parameters())
            dual_loss = dual_objective(m.lagrange(batch['z'], batch['proprio']), rho, cfg.get('epsilon', .3))
            optimize(dual_loss, self.optimizers['multiplier'], m.multiplier.parameters())
            m.trained_steps += 1
            result.update(actor_loss=float(actor_loss.detach()), dual_loss=float(dual_loss.detach()), rho=float(rho.mean().detach()))
        return result

    def on_new_transitions(self, sample_replay, sample_human, new_human=0, batch_size=256):
        """Call once per genuinely new chunk, not once per offline replay epoch.

        Samplers return tensor dictionaries. Replay includes human transitions;
        human buffer is their subset, matching the paper's 1:1 sampling rule.
        """
        if batch_size % 2:
            raise ValueError('1:1 replay/human sampling requires even batch size')
        self.new_human += new_human
        result = []
        for _ in range(5):
            replay, human = sample_replay(batch_size//2), sample_human(batch_size//2)
            batch = {k: torch.cat([replay[k], human[k]]) for k in replay}
            result.append(self.update(batch))
        while self.new_human >= 100:
            for _ in range(12):
                self.correction_step(sample_human(batch_size))
            self.new_human -= 100
        return result
