"""Save and restore learner, milestone and imitation checkpoints.

The payload formats are unchanged from the original scripts, so existing checkpoints
still load: plain dicts of tensors, configs, counters and RNG states.
"""
import random
from pathlib import Path

import numpy as np
import torch

from pokemon_agent.files import atomic_torch_save
from pokemon_agent.model.serialization import load_weights, read_checkpoint

LEARNER_PROJECT = "ps-ppo-15token"
MILESTONE_PROJECT = "ps-ppo-15token-milestone"
IMITATION_PROJECT = "ps-ppo-15token-imitation"
ARCHITECTURE = "readme_15token"


def learner_checkpoint_name(update_idx: int) -> str:
    return f"learner_update_{update_idx:09d}.pt"


def prune_checkpoints(directory, keep_last: int) -> None:
    """Delete all but the newest keep_last learner checkpoints (0 keeps everything)."""
    if keep_last > 0:
        for old in sorted(Path(directory).glob("learner_update_*.pt"))[:-keep_last]:
            old.unlink()


def save_learner_checkpoint(path, net, opt, sched, run_cfg, dataset, *, update_idx, total_steps,
        total_episodes, training_totals):
    """Everything a same-phase resume needs, including not-yet-trained decisions and RNG states."""
    cuda = run_cfg.learner.device.startswith("cuda")
    payload = dict(project=LEARNER_PROJECT, architecture=ARCHITECTURE, format_version=1,
        model=net.state_dict(), model_config=net.reproduction_config,
        run_cfg=run_cfg.as_dict(), phase=run_cfg.learner.mode, optimizer=opt.state_dict(),
        scheduler=sched.state_dict(), update_idx=update_idx, total_steps=total_steps,
        total_episodes=total_episodes, torch_rng=torch.get_rng_state(),
        training_totals=training_totals,
        cuda_rng=torch.cuda.get_rng_state_all() if cuda else None,
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        pending=dataset.tensorize() if len(dataset) else None)
    path = atomic_torch_save(payload, path)
    prune_checkpoints(path.parent, run_cfg.learner.keep_last)
    return str(path.resolve())


def check_optimizer_settings(checkpoint, learner_cfg) -> None:
    """Same-phase resume restores saved rates; refuse settings that would be silently ignored."""
    saved = checkpoint.get("run_cfg", {}).get("learner", {})
    # Checkpoints written before these settings existed used these values.
    saved = {"lr_schedule": "historical", "ppo_lr_multipliers": "current", **saved}
    names = ["lr", "lr_schedule"] + (["ppo_lr_multipliers"] if learner_cfg.mode.startswith("ppo") else [])
    changed = {name: (saved.get(name), getattr(learner_cfg, name)) for name in names
        if saved.get(name) != getattr(learner_cfg, name)}
    if changed:
        raise ValueError(f"Resuming {learner_cfg.mode} would keep the checkpoint's optimizer settings, "
            f"not the requested ones (saved, requested): {changed}. Pass the saved settings to "
            "resume this run, or start a new run from an imitation or warmup checkpoint.")


def restore_learner_checkpoint(path, net, opt, sched, dataset, learner_cfg):
    """Load a checkpoint's weights; for the same phase, also its optimizer, counters, pending data and RNG.

    Returns the restored counters, or None when the checkpoint is another phase
    (or a milestone, imitation or upgrade file) and only its weights are used.
    """
    checkpoint = read_checkpoint(path)
    load_weights(net, checkpoint)
    same_phase = checkpoint.get("project") == LEARNER_PROJECT and checkpoint.get("phase") == learner_cfg.mode
    if not same_phase:
        return None
    check_optimizer_settings(checkpoint, learner_cfg)
    opt.load_state_dict(checkpoint["optimizer"])
    sched.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("pending") is not None:
        dataset.add_steps(*checkpoint["pending"])
    torch.set_rng_state(checkpoint["torch_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    if learner_cfg.device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return dict(update_idx=checkpoint["update_idx"], total_steps=checkpoint["total_steps"],
        total_episodes=checkpoint["total_episodes"], training_totals=checkpoint.get("training_totals", {}))


def save_milestone_if_due(directory, net, run_cfg, *, last_milestone, total_steps, update_idx, total_episodes):
    """Keep a permanent policy-only copy each time new states cross a milestone.

    Regular checkpoints rotate (keep_last); milestones are never deleted and load
    directly in evaluation. A later run started from one gets a fresh optimizer.
    Returns (milestones reached, saved path or None).
    """
    every = run_cfg.learner.milestone_every_states
    if not every:
        return last_milestone, None
    reached = total_steps // every
    if reached <= last_milestone:
        return last_milestone, None
    path = Path(directory) / "milestones" / f"states-{reached*every:012d}.pt"
    payload = dict(project=MILESTONE_PROJECT, architecture=ARCHITECTURE, format_version=1,
        phase=run_cfg.learner.mode, model=net.state_dict(), model_config=net.reproduction_config,
        milestone_states=reached*every, total_steps=total_steps, update_idx=update_idx,
        total_episodes=total_episodes, run_cfg=run_cfg.as_dict())
    atomic_torch_save(payload, path)
    return reached, str(path.resolve())


def save_imitation_checkpoint(path, model, architecture: dict, *, epoch, next_shard, best_accuracy,
        validation_history, optimizer=None, scheduler=None, device="cpu"):
    """best.pt holds weights only; last.pt (optimizer given) can also resume the shard loop."""
    payload = dict(project=IMITATION_PROJECT, architecture=ARCHITECTURE, phase="imitation",
        model=model.state_dict(), model_config=architecture, epoch=epoch, next_shard=next_shard,
        best_accuracy=best_accuracy, validation_history=validation_history)
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)
    return atomic_torch_save(payload, path)


def restore_imitation_checkpoint(saved, model, optimizer, scheduler, device="cpu"):
    """Resume from last.pt; returns (epoch, next_shard, best_accuracy, validation_history)."""
    load_weights(model, saved)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["torch_rng"])
    if device.startswith("cuda") and saved.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["epoch"], saved["next_shard"], saved["best_accuracy"], saved["validation_history"]
