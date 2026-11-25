"""Action distributions under the legal-action mask: masking, sampling, log-probabilities."""
from __future__ import annotations
import logging
from typing import Tuple
import torch
from torch.distributions import Categorical
logger = logging.getLogger(__name__)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a boolean mask to logits, setting invalid actions to a large negative value.
    
    Includes a safety valve to prevent NaN outputs if an entirely zero mask is provided
    by forcing at least one valid index.
    """
    mask_sum = mask.sum(dim=-1)
    if (mask_sum == 0).any():
        bad_indices = (mask_sum == 0).nonzero(as_tuple=True)[0]
        mask = mask.clone()
        mask[bad_indices, 0] = 1.0
        logger.warning(f"Zero mask detected at batch indices {bad_indices.tolist()}. Forced index 0.")

    m = (mask > 0.5).to(torch.bool)
    return logits.masked_fill(~m, -1e4)


@torch.no_grad()
def masked_sample(
    logits: torch.Tensor, 
    mask: torch.Tensor, 
    greedy: bool = False, 
    temp: float = 1.0,
    top_p: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples actions using temperature scaling and optional Nucleus (Top-P) filtering.
    
    Args:
        logits: Raw action scores.
        mask: Valid action mask.
        greedy: If True, returns argmax.
        temp: Temperature (>1.0 increases entropy, <1.0 decreases it).
        top_p: Nucleus sampling threshold.
    """
    ml_pure = masked_logits(logits, mask)
    dist_pure = Categorical(logits=ml_pure)
    
    # Scale logits for exploration
    ml_explore = masked_logits(logits / max(temp, 1e-4), mask)
     
    if greedy:
        a = torch.argmax(ml_pure, dim=-1)
        return a, dist_pure.log_prob(a), torch.zeros_like(a, dtype=torch.float32)
    
    if 0.0 < top_p < 1.0:
        probs = torch.softmax(ml_explore, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens outside the nucleus
        sorted_to_remove = cumulative_probs > top_p
        sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
        sorted_to_remove[..., 0] = False
        
        indices_to_remove = sorted_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_to_remove)
        ml_explore = ml_explore.masked_fill(indices_to_remove, float('-inf'))
    
    dist_explore = Categorical(logits=ml_explore)
    a = dist_explore.sample()
    return a, dist_explore.log_prob(a), dist_explore.entropy()


def masked_logprob_entropy(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates log probabilities and entropy for a specific set of actions under a mask."""
    ml = masked_logits(logits, mask)
    dist = Categorical(logits=ml)
    return dist.log_prob(actions), dist.entropy()
