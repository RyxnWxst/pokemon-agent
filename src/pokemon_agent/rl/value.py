"""Distributional value head helpers: two-hot targets, value loss, value variance."""
from __future__ import annotations
import torch


def twohot_targets(x: torch.Tensor, *, v_min: float, v_max: float, v_bins: int) -> torch.Tensor:
    """
    Encodes scalar values into a two-hot distribution over uniform bins.
    
    Used for distributional value prediction to reduce variance and improve 
    learning stability in reinforcement learning.
    """
    x = x.clamp(v_min, v_max)
    scale = (v_bins - 1) / (v_max - v_min)
    f = (x - v_min) * scale
    i0 = torch.floor(f).long()
    i1 = torch.clamp(i0 + 1, max=v_bins - 1)

    w1 = (f - i0.float())
    w0 = 1.0 - w1

    # Edge case: exactly at the maximum bin
    w0 = torch.where(i0 == i1, torch.ones_like(w0), w0)
    w1 = torch.where(i0 == i1, torch.zeros_like(w1), w1)

    t = torch.zeros((x.shape[0], v_bins), device=x.device, dtype=torch.float32)
    t.scatter_add_(1, i0.view(-1, 1), w0.view(-1, 1))
    t.scatter_add_(1, i1.view(-1, 1), w1.view(-1, 1))
    return t


def dist_value_loss(v_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss between predicted value logits and target distributions."""
    logp = torch.log_softmax(v_logits, dim=-1)
    return -(target_dist * logp).sum(dim=-1).mean()


def value_variance(value_logits: torch.Tensor, values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Variance of the critic's return distribution around its mean, per state."""
    probabilities = torch.softmax(value_logits, dim=-1)
    return (probabilities*(support-values[:, None]).square()).sum(-1)
