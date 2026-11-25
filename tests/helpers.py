"""Shared test fixtures: the tiny architecture, recorded observations, and a learner without Ray."""
from dataclasses import replace
import json
from pathlib import Path
import torch

from pokemon_agent.config import LearnerConfig, ModelConfig, RunConfig
from pokemon_agent.distributed.learner import LearnerActor
from pokemon_agent.model.transformer import make_model
from pokemon_agent.rl.buffer import AsyncEpisodeDataset
from pokemon_agent.rl.distributions import masked_logprob_entropy
from pokemon_agent.training.optimizer import make_optimizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def tiny_architecture() -> ModelConfig:
    """A small model with the real observation layout, for fast tests."""
    return ModelConfig(**json.loads((FIXTURES/"tiny_model.json").read_text()))


def load_fixture(name="battle-observations.pt"):
    return torch.load(FIXTURES/name, weights_only=True)


def fixture_dataset(model, obs, actions):
    """The recorded decisions with the model's own log-probabilities and values and made-up advantages."""
    with torch.no_grad():
        logits, _, values, *_ = model(obs)
        start, end = model.unpacker.offsets["action_mask"]
        logp, _ = masked_logprob_entropy(logits, obs[:, start:end], actions)
    n = len(actions)
    dataset = AsyncEpisodeDataset(14, "cpu")
    dataset.add_steps(obs, actions, logp, values, torch.linspace(-1, 1, n),
        torch.linspace(1, -1, n), torch.zeros(n), torch.arange(n))
    return dataset


def bare_learner(directory, architecture, cfg=None, **settings):
    """A LearnerActor with a model, optimizer and empty dataset but no Ray, queue or background task."""
    learner = LearnerActor.__new__(LearnerActor)
    cfg = cfg or replace(LearnerConfig(), device="cpu", mode="ppo")
    learner.cfg = replace(cfg, ckpt_dir=str(directory), **settings)
    learner.run_cfg = replace(RunConfig.default(), model=architecture, learner=learner.cfg)
    learner.net = make_model(architecture)
    learner.opt, learner.sched = make_optimizer(learner.net, learner.cfg)
    learner.dataset = AsyncEpisodeDataset(14, "cpu")
    learner.update_idx = learner.total_steps = learner.total_episodes = 0
    learner.directory = Path(directory)
    return learner
