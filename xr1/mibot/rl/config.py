from dataclasses import dataclass


@dataclass(frozen=True)
class LWDConfig:
    stage: str = "offline"
    atoms: int = 201
    v_min: float = -0.1
    v_max: float = 1.1
    gamma: float = 0.9999
    tau_offline: float = 0.6
    tau_online: float = 0.9
    entropy_coefficient: float = 0.3
    # The paper does not publish these clipping bounds; these are local defaults.
    tau_min: float = 0.0
    tau_max: float = 1.0
    ema_rate: float = 0.005
    temperature: float = 2.0
    flow_steps: int = 5
    actor_lr: float = 2e-5
    critic_lr: float = 5e-4
    actor_weight_decay: float = 0.01
    grad_clip: float = 1.0
    total_steps: int = 100000

    def __post_init__(self):
        if self.stage not in ("offline", "online"):
            raise ValueError("stage must be offline or online")
        if self.atoms < 2 or self.v_min >= self.v_max:
            raise ValueError("invalid categorical support")
        if not 0 < self.gamma <= 1 or not 0 < self.ema_rate <= 1:
            raise ValueError("invalid discount or EMA rate")
        if not 0 <= self.tau_min <= self.tau_max <= 1:
            raise ValueError("invalid quantile bounds")
        if not 0 <= self.tau_offline <= 1 or not 0 <= self.tau_online <= 1:
            raise ValueError("invalid base quantile")
        if self.entropy_coefficient < 0 or self.temperature <= 0:
            raise ValueError("invalid entropy coefficient or temperature")
        if self.flow_steps < 1 or self.total_steps < 1:
            raise ValueError("step counts must be positive")
        if min(self.actor_lr, self.critic_lr, self.grad_clip) <= 0:
            raise ValueError("learning rates and gradient clipping must be positive")

    @property
    def tau_base(self):
        return self.tau_offline if self.stage == "offline" else self.tau_online
