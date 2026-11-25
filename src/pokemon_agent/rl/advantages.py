"""Advantages: GAE per episode, applied to packed episodes, and normalized per update."""
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch


@torch.no_grad()
def gae_from_episode(
    rewards: torch.Tensor,     # [T]
    values: torch.Tensor,      # [T]
    dones: torch.Tensor,       # [T] (1 at terminal step)
    gamma: float,
    lam: float,
    last_value: float = 0.0,   # terminal -> 0
    variances: Optional[torch.Tensor] = None, # [T] State-conditional variance
    lam_min: float = 0.55,
    lam_max: float = 0.95,
    lam_eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes GAE for one episode. If variances are provided, computes state-aware 
    dynamic lambda per timestep to balance bias and variance dynamically.
    """
    T = rewards.shape[0]
    adv = torch.zeros((T,), device=rewards.device)
    gae = torch.tensor(0.0, device=rewards.device)

    # ---------------------------------------------------------
    # Branch A: Dynamic Lambda calculation using per-state Variance
    # ---------------------------------------------------------
    if variances is not None:
        # Precompute 1-step TD Errors for the entire episode
        next_values = torch.cat([values[1:], torch.tensor([last_value], device=values.device)])
        next_nonterminal = 1.0 - dones
        
        deltas = rewards + gamma * next_values * next_nonterminal - values
        
        # Vectorized dynamic lambda calculation
        delta_sq = deltas ** 2
        dynamic_lambda = delta_sq / (delta_sq + variances + lam_eps)
        dynamic_lambda = torch.clamp(dynamic_lambda, min=lam_min, max=lam_max)

        for t in reversed(range(T)):
            gae = deltas[t] + gamma * dynamic_lambda[t] * next_nonterminal[t] * gae
            adv[t] = gae

    # ---------------------------------------------------------
    # Branch B: Standard Static GAE
    # ---------------------------------------------------------
    else:
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = torch.tensor(last_value, device=rewards.device) if (t == T - 1) else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            adv[t] = gae

    ret = adv + values
    return adv, ret


def episode_advantages(rewards, values, dones, variances, lengths: Sequence[int], cfg) -> Tuple[list, list]:
    """Run gae_from_episode over each episode of a packed batch; returns per-episode advantages and returns."""
    advantages, returns = [], []
    start = 0
    for length in lengths:
        end = start + int(length)
        adv, ret = gae_from_episode(rewards[start:end], values[start:end], dones[start:end],
            cfg.gamma, cfg.gae_lambda,
            variances=variances[start:end] if cfg.use_dynamic_lambda else None,
            lam_min=cfg.dynamic_lambda_min, lam_max=cfg.dynamic_lambda_max,
            lam_eps=cfg.dynamic_lambda_eps)
        advantages.append(adv)
        returns.append(ret)
        start = end
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, clamp: float = 10.) -> torch.Tensor:
    """Whole-update normalization: zero mean, unit standard deviation, clamped to +-clamp."""
    return ((advantages-advantages.mean()) / (advantages.std(unbiased=False)+1e-8)).clamp(-clamp, clamp)
