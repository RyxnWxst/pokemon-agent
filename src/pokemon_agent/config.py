"""Configuration for the standalone, stateless 15-token project."""
from dataclasses import asdict, dataclass, field

from pokemon_agent.paths import checkpoints_dir
from pokemon_agent.vendor.historical_config import ModelConfig as HistoricalModelConfig


def default_device() -> str:
    """CUDA when PyTorch can see a GPU, otherwise the CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ModelConfig(HistoricalModelConfig):
    """Historical architecture plus explicit observation-adapter versioning.

    move_category_inputs restores the README-era Physical/Special/Status move
    one-hot at historical move-scalar indices 3-5. The current observation
    schema stores that one-hot outside the slice read by the historical move
    encoder. Checkpoints saved before this field existed load as False.
    """
    move_category_inputs: bool = True


# PPO learning-rate multipliers by parameter zone. "historical" is de8cc69's
# (1, 1, 1); "current" is the later upstream (backbone 0.5, policy 1, value 2).
PPO_LR_MULTIPLIERS = {
    "historical": dict(backbone=1., pi=1., v=1.),
    "current": dict(backbone=.5, pi=1., v=2.),
}


@dataclass(frozen=True)
class EnvConfig:
    battle_format: str = "gen9randombattle"
    act_dim: int = 14


@dataclass(frozen=True)
class RolloutConfig:
    target_concurrent_battles: int = 8
    rooms_per_pair: int = 8
    infer_timeout_s: float = 30.
    open_timeout: float = 30.
    infer_min_batch: int = 1
    infer_max_batch: int = 2048
    infer_wait_ms: float = 2.
    infer_max_pending: int = 20000
    learn_min_episodes: int = 1
    learn_max_episodes: int = 16
    learn_wait_ms: float = 1.
    learn_max_pending_episodes: int = 32
    learn_max_pending_batches: int = 5
    # Finished episodes a worker may hold while the learner is busy. 0 keeps the
    # original bound (learn_max_pending_batches * learn_max_episodes).
    learn_queue_episodes: int = 0


@dataclass(frozen=True)
class InferenceConfig:
    device: str = field(default_factory=default_device)


@dataclass(frozen=True)
class LearnerConfig:
    device: str = field(default_factory=default_device)
    mode: str = "ppo"
    gamma: float = .9999
    gae_lambda: float = .75
    use_dynamic_lambda: bool = False
    dynamic_lambda_min: float = .55
    dynamic_lambda_max: float = .95
    dynamic_lambda_eps: float = 1e-8
    v_min: float = -1.6
    v_max: float = 1.6
    v_bins: int = 51
    temp_start: float = 1.
    temp_end: float = 1.
    temp_total_steps: int = 250_000_000
    lr: float = 1e-4
    lr_schedule: str = "historical"
    lr_warmup_steps: int = 1000
    lr_hold_steps: int = 20000
    lr_total_steps: int = 500000
    weight_decay: float = .01
    # README-era (de8cc69) PPO recipe: 32,768-state updates, 4,096-state optimizer
    # minibatches, three passes, KL stopping only after the first full pass, and
    # equal learning rates for backbone, policy and value. Microbatches only bound
    # GPU memory; gradients are accumulated to the full minibatch.
    update_epochs: int = 3
    minibatch_size: int = 4096
    microbatch_size: int = 512
    kl_stop_rule: str = "historical"
    ppo_lr_multipliers: str = "historical"
    clip_coef: float = .2
    ent_coef: float = .02
    vf_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float = .02
    steps_per_update: int = 32768
    ckpt_dir: str = field(default_factory=lambda: str(checkpoints_dir() / "ppo"))
    save_every_updates: int = 25
    keep_last: int = 3
    # Policy-only copies kept permanently under <ckpt_dir>/milestones; 0 disables.
    milestone_every_states: int = 10_000_000
    resume: bool = True
    resume_path: str | None = None
    max_updates: int = 0
    max_total_steps: int = 0

    def __post_init__(self):
        if self.mode not in ("imitation", "warmup", "ppo", "ppo_vs_teacher"):
            raise ValueError(f"Unsupported 15-token mode: {self.mode}")
        if min(self.lr, self.minibatch_size, self.update_epochs, self.steps_per_update) <= 0:
            raise ValueError("Learning rate and batch/update sizes must be positive")
        if self.max_total_steps < 0 or self.max_updates < 0 or self.milestone_every_states < 0:
            raise ValueError("Stop limits and milestone interval cannot be negative")
        if self.microbatch_size < 0:
            raise ValueError("Microbatch size cannot be negative")
        if self.kl_stop_rule not in ("minibatch", "historical"):
            raise ValueError("KL stopping rule must be minibatch or historical")
        if self.lr_schedule not in ("historical", "constant"):
            raise ValueError("Unknown learning-rate schedule")
        if self.ppo_lr_multipliers not in PPO_LR_MULTIPLIERS:
            raise ValueError("PPO learning-rate multipliers must be historical or current")
        if self.temp_start != 1 or self.temp_end != 1:
            raise ValueError("Use temperature 1 so PPO behavior and update log probabilities agree")

    def ppo_kwargs(self):
        names = ("update_epochs", "minibatch_size", "clip_coef", "ent_coef", "vf_coef",
                 "max_grad_norm", "target_kl", "microbatch_size", "kl_stop_rule")
        return {name: getattr(self, name) for name in names}


@dataclass(frozen=True)
class RewardConfig:
    terminal_win: float = 1.
    terminal_loss: float = -1.
    use_faint_reward: bool = True
    faint_self: float = -.1
    faint_opp: float = .1


@dataclass(frozen=True)
class RunConfig:
    matmul_precision: str = "highest"
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def default(cls):
        return cls()

    def as_dict(self):
        return asdict(self)
