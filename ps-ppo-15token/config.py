"""Configuration for the standalone, stateless 15-token project."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import torch
from vendor.historical_config import ModelConfig as HistoricalModelConfig

ROOT = Path(__file__).resolve().parent


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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LearnerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
    # "historical": warmup, hold, slow decay by optimizer step (de8cc69).
    # "constant": fixed rate. "linear": warmup, then decay linearly with new
    # states to lr_final_fraction * lr at max_total_steps (needs a state target).
    lr_schedule: str = "historical"
    lr_final_fraction: float = 0.
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
    ckpt_dir: str = str(ROOT / "checkpoints/ppo")
    save_every_updates: int = 25
    keep_last: int = 3
    # Policy-only copies kept permanently under <ckpt_dir>/milestones; 0 disables.
    milestone_every_states: int = 10_000_000
    resume: bool = True
    resume_path: str | None = None
    max_updates: int = 0
    max_total_steps: int = 0
    # Who the trained policy plays: "self" (both sides are the current policy) or
    # "heuristic" (the corrected SimpleHeuristicsPlayer; only our side is trained).
    # Mode ppo_vs_teacher always uses the heuristic.
    opponent: str = "self"
    # Retained for worker/report compatibility; replay is disabled in this project.
    teacher_probability: float = .5
    replay_steps: int = 0

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
        if self.lr_schedule not in ("historical", "constant", "linear"):
            raise ValueError("Unknown learning-rate schedule")
        if self.lr_schedule == "linear" and self.max_total_steps <= 0:
            raise ValueError("The linear learning-rate schedule decays over a state target; set max_total_steps")
        if not 0 <= self.lr_final_fraction <= 1:
            raise ValueError("lr_final_fraction must be between 0 and 1")
        if self.opponent not in ("self", "heuristic"):
            raise ValueError("Opponent must be self or heuristic")
        if self.opponent == "heuristic" and self.mode == "imitation":
            raise ValueError("Imitation does not play an opponent")
        if self.ppo_lr_multipliers not in PPO_LR_MULTIPLIERS:
            raise ValueError("PPO learning-rate multipliers must be historical or current")
        if self.temp_start != 1 or self.temp_end != 1:
            raise ValueError("Use temperature 1 so PPO behavior and update log probabilities agree")

    @property
    def heuristic_opponent(self):
        return self.mode == "ppo_vs_teacher" or self.opponent == "heuristic"

    def get_temp(self, total_steps):
        return 1.

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
    # "fixed" reads the fainted flag. "legacy" reproduces runs made before the fix,
    # which read the "ability unknown" flag: +0.1 when an opponent Pokemon with an
    # unknown ability appeared, never a penalty for our own faints. See rewards.py.
    faint_index: str = "fixed"
    # Potential-based HP-lead shaping coefficient c; 0 disables. See rewards.py.
    hp_shaping: float = 0.

    def __post_init__(self):
        if self.faint_index not in ("fixed", "legacy"):
            raise ValueError("faint_index must be fixed or legacy")
        if self.hp_shaping < 0:
            raise ValueError("hp_shaping cannot be negative")


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

    def make_model(self):
        from model import make_model
        return make_model(self.model)

    def as_dict(self):
        return asdict(self)
