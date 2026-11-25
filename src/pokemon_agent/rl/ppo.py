"""The update step: imitation cross-entropy, critic-only warmup, or clipped PPO with a distributional critic."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from pokemon_agent.rl.distributions import masked_logprob_entropy
from pokemon_agent.rl.value import dist_value_loss, twohot_targets


@dataclass
class PPOUpdateStats:
    """Statistics container for a single PPO update batch."""
    approx_kl: float
    clip_frac: float
    entropy: float
    v_loss: float
    pg_loss: float
    total_loss: float
    n_mb: int
    n_microbatches: int = 0
    samples_processed: int = 0
    first_pass_samples: int = 0
    first_pass_fraction: float = 0.
    epochs_completed: int = 0
    early_stopped: bool = False
    stop_kl: float | None = None


def imitation_loss(logits: torch.Tensor, actions: torch.Tensor, label_smoothing: float = .1) -> torch.Tensor:
    """Cross-entropy toward the teacher's actions. Uses the raw logits, so smoothing covers all 14 actions."""
    return F.cross_entropy(logits.float(), actions, label_smoothing=label_smoothing)


def clipped_policy_loss(logits, mask, actions, old_logp, advantages, clip_coef: float = .2):
    """PPO's clipped surrogate. Returns (policy_loss, entropy, approx_kl, signed_kl, clip_frac).

    approx_kl is the (ratio - 1) - log(ratio) estimator used for logging and the
    minibatch stopping rule; signed_kl is mean(old_logp - new_logp), used by the
    historical rule.
    """
    logp, entropies = masked_logprob_entropy(logits.float(), mask, actions)
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    policy_loss = torch.maximum(-advantages*ratio,
        -advantages*ratio.clamp(1-clip_coef, 1+clip_coef)).mean()
    entropy = entropies.mean()
    approx_kl = ((ratio-1)-log_ratio).mean()
    signed_kl = -log_ratio.mean()
    clip_frac = ((ratio-1).abs() > clip_coef).float().mean()
    return policy_loss, entropy, approx_kl, signed_kl, clip_frac


def kl_should_stop(rule: str, epoch: int, batch_kl: float, epoch_signed_kl: float,
                   epoch_batches: int, target_kl: float | None) -> float | None:
    """The KL value that ends an update early, or None to keep going.

    "minibatch" checks each minibatch's approx KL, even in the first pass.
    "historical" (de8cc69) waits until the second pass, then checks the running
    mean of the signed KL over that pass. Both stop above 1.5 x target_kl.
    """
    check = batch_kl if rule == "minibatch" else epoch_signed_kl/epoch_batches
    eligible = rule == "minibatch" or epoch > 0
    if target_kl and eligible and check > 1.5*target_kl:
        return check
    return None


def ppo_update(net, opt, dataset, scheduler=None, mode="ppo", **cfg):
    """Historical imitation CE or clipped PPO with distributional value loss."""
    if mode not in ("imitation", "warmup", "ppo", "ppo_vs_teacher"):
        raise ValueError(f"Unsupported mode: {mode}")
    if not len(dataset):
        raise ValueError("Cannot update on an empty rollout")
    device = next(net.parameters()).device
    sums = dict(approx_kl=0., clip_frac=0., entropy=0., v_loss=0., pg_loss=0., total_loss=0.)
    batches = 0
    microbatches = samples_processed = first_pass_samples = epochs_completed = 0
    stop = False
    stop_kl = None
    stop_rule = cfg.get("kl_stop_rule", "minibatch")
    if stop_rule not in ("minibatch", "historical"):
        raise ValueError(f"Unsupported KL stopping rule: {stop_rule}")
    net.train()
    for epoch in range(cfg.get("update_epochs", 1)):
        epoch_signed_kl = 0.
        epoch_batches = epoch_samples = 0
        for batch in dataset.iter_minibatches(
                cfg.get("minibatch_size", 64), shuffle_episodes=True):
            size = len(batch[1])
            micro_size = cfg.get("microbatch_size", 0) or size
            if micro_size < 1:
                raise ValueError("Microbatch size must be positive, or zero for a full minibatch")
            batch_values = {name: 0. for name in sums}
            signed_kl = 0.
            opt.zero_grad(set_to_none=True)
            for offset in range(0, size, micro_size):
                obs, actions, old_logp, advantages, returns = [batch[index][offset:offset+micro_size].to(device)
                    for index in (0, 1, 2, 4, 5)]
                logits, value_logits, *_ = net(obs)
                start, end = net.unpacker.offsets["action_mask"]
                mask = obs[:, start:end]
                if bool((mask.gather(1, actions[:, None]) < .5).any()):
                    raise ValueError("Recorded action is outside the legal action mask")
                zero = torch.zeros((), device=device)
                signed = zero
                if mode == "imitation":
                    policy_loss = imitation_loss(logits, actions)
                    value_loss = entropy = kl = clip = zero
                    loss = policy_loss
                else:
                    targets = twohot_targets(returns, v_min=cfg.get("v_min", -1.6),
                        v_max=cfg.get("v_max", 1.6), v_bins=cfg.get("v_bins", 51))
                    value_loss = dist_value_loss(value_logits, targets)
                    if mode == "warmup":
                        policy_loss = entropy = kl = clip = zero
                        loss = value_loss
                    else:
                        policy_loss, entropy, kl, signed, clip = clipped_policy_loss(
                            logits, mask, actions, old_logp, advantages, cfg.get("clip_coef", .2))
                        loss = policy_loss + cfg.get("vf_coef", .5)*value_loss - cfg.get("ent_coef", .02)*entropy
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite loss; refusing to update model")
                weight = len(actions)/size
                (loss*weight).backward()
                values = dict(approx_kl=kl, clip_frac=clip, entropy=entropy, v_loss=value_loss,
                    pg_loss=policy_loss, total_loss=loss)
                for name, value in values.items():
                    batch_values[name] += float(value.detach())*weight
                signed_kl += float(signed.detach())*weight
                microbatches += 1
            nn.utils.clip_grad_norm_(net.parameters(), cfg.get("max_grad_norm", .5), error_if_nonfinite=True)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            for name, value in batch_values.items():
                sums[name] += value
            batches += 1
            samples_processed += size
            epoch_samples += size
            if epoch == 0:
                first_pass_samples += size
            epoch_signed_kl += signed_kl
            epoch_batches += 1
            if mode.startswith("ppo"):
                stop_kl = kl_should_stop(stop_rule, epoch, batch_values["approx_kl"], epoch_signed_kl,
                    epoch_batches, cfg.get("target_kl"))
                if stop_kl is not None:
                    stop = True
                    break
        if epoch_samples == len(dataset):
            epochs_completed += 1
        if stop:
            break
    return PPOUpdateStats(**{name: value/batches for name,value in sums.items()}, n_mb=batches,
        n_microbatches=microbatches, samples_processed=samples_processed,
        first_pass_samples=first_pass_samples, first_pass_fraction=first_pass_samples/len(dataset),
        epochs_completed=epochs_completed, early_stopped=stop, stop_kl=stop_kl)
