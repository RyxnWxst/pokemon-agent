"""Offline imitation over manifest shards: choose the architecture, train, score and save."""
import json
from pathlib import Path
import random

import torch

from pokemon_agent.config import ModelConfig
from pokemon_agent.files import atomic_json
from pokemon_agent.model.serialization import model_config
from pokemon_agent.rl.buffer import make_dataset
from pokemon_agent.rl.ppo import ppo_update
from pokemon_agent.training.checkpoints import save_imitation_checkpoint
from pokemon_agent.training.datasets import assert_disjoint
from pokemon_agent.training.metrics import action_metrics, combine_metrics


def _load_shard(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_architecture(saved=None, initial=None, model_config_path=None) -> ModelConfig:
    """A resumed run keeps last.pt's architecture, then --init-checkpoint's, then --model-config, else the default."""
    if saved:
        return model_config(saved)
    if initial:
        return model_config(initial)
    if model_config_path:
        return ModelConfig(**json.loads(Path(model_config_path).read_text()))
    return ModelConfig()


def score_initial_weights(model, validation, history, *, directory, architecture: dict, report, status_path):
    """Score the starting weights so best.pt is never worse than the initialization."""
    metrics = action_metrics(model, validation)
    history.append(dict(epoch=0, shard=0, metrics=metrics, loss=None))
    best = metrics["accuracy"]
    save_imitation_checkpoint(Path(directory)/"best.pt", model, architecture, epoch=0, next_shard=0,
        best_accuracy=best, validation_history=history)
    report.update(initial_validation=metrics, best_validation_accuracy=best)
    atomic_json(status_path, report)
    print(json.dumps(dict(epoch=0, heldout_accuracy=best)), flush=True)
    return best


def train_imitation(model, optimizer, scheduler, cfg, shard_paths, validation, heldout_hashes, *, epochs,
        start_epoch, next_shard, best, history, directory, architecture: dict, report, status_path):
    """One imitation pass per shard; shards are shuffled each epoch with seed 7100+epoch.

    After every shard: score the held-out set, save best.pt when accuracy improves,
    and always save last.pt with the optimizer so the run can resume mid-epoch.
    Returns the best held-out accuracy.
    """
    directory = Path(directory)
    for epoch in range(start_epoch, epochs+1):
        order = list(range(len(shard_paths)))
        random.Random(7100+epoch).shuffle(order)
        for position in range(next_shard if epoch == start_epoch else 0, len(order)):
            episodes = _load_shard(shard_paths[order[position]])
            assert_disjoint(episodes, heldout_hashes)
            dataset = make_dataset(episodes)
            del episodes
            report.update(epoch=epoch, shard=position+1, total_shards=len(shard_paths))
            atomic_json(status_path, report)
            stats = ppo_update(model, optimizer, dataset, scheduler=scheduler, mode="imitation",
                **cfg.ppo_kwargs(), v_min=cfg.v_min, v_max=cfg.v_max, v_bins=cfg.v_bins)
            del dataset
            metrics = action_metrics(model, validation)
            history.append(dict(epoch=epoch, shard=position+1, metrics=metrics, loss=stats.pg_loss))
            if metrics["accuracy"] > best:
                best = metrics["accuracy"]
                save_imitation_checkpoint(directory/"best.pt", model, architecture, epoch=epoch,
                    next_shard=position+1, best_accuracy=best, validation_history=history)
            save_imitation_checkpoint(directory/"last.pt", model, architecture, epoch=epoch,
                next_shard=position+1, best_accuracy=best, validation_history=history,
                optimizer=optimizer, scheduler=scheduler, device=cfg.device)
            report.update(best_validation_accuracy=best, last_validation=metrics)
            atomic_json(status_path, report)
            print(json.dumps(dict(epoch=epoch, shard=position+1, heldout_accuracy=metrics["accuracy"])), flush=True)
    return best


def score_training_shards(model, shard_paths) -> dict:
    """Final accuracy over all training shards, one shard in memory at a time."""
    scores = []
    for path in shard_paths:
        episodes = _load_shard(path)
        dataset = make_dataset(episodes)
        del episodes
        scores.append(action_metrics(model, dataset))
        del dataset
    return combine_metrics(scores)
