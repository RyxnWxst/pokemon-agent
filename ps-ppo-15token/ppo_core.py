"""Stateless 15-token observation, rollout-buffer, and PPO utilities."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Iterator, Final, List
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ObservationUnpacker(nn.Module):
    """
    Slices and reshapes flat observation tensors into structured components.
    
    Expects metadata containing 'offsets' and dynamic dimensions from the 
    ObservationAssembler.
    """
    def __init__(self, meta: dict):
        super().__init__()
        self.meta = meta
        self.offsets = meta["offsets"]
        self.dim_body = meta["dim_pokemon_body"]
        self.dim_move_sc = meta["dim_move_scalars"] // 4
        self.dim_transition_sc = meta["dim_transition_scalars"]
        self.transition_map = meta["feature_map"]["transition"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {name: x[:, start:end] for name, (start, end) in self.offsets.items()}
        
        # Reshape structured fields
        out["pokemon_body"] = out["pokemon_body"].reshape(-1, 12, self.dim_body)
        out["pokemon_ids"]  = out["pokemon_ids"].reshape(-1, 12, 2).to(torch.long)
        out["ability_ids"]  = out["ability_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_ids"]     = out["move_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_scalars"] = out["move_scalars"].reshape(-1, 12, 4, self.dim_move_sc)
        out["transition_ids"] = out["transition_ids"].reshape(-1, self.meta["dim_transition_ids"]).to(torch.long)
        out["transition_scalars"] = out["transition_scalars"].reshape(-1, self.dim_transition_sc)
        
        tmap = self.transition_map
        out["transition_move_ids"] = out["transition_ids"][:, tmap["move_ids"][0]:tmap["move_ids"][1]]
        out["transition_pokemon_ids"] = out["transition_ids"][:, tmap["pokemon_ids"][0]:tmap["pokemon_ids"][1]]
        out["transition_ability_ids"] = out["transition_ids"][:, tmap["ability_ids"][0]:tmap["ability_ids"][1]]
        out["transition_item_ids"] = out["transition_ids"][:, tmap["item_ids"][0]:tmap["item_ids"][1]]
        
        return out


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


class AsyncEpisodeDataset:
    def __init__(self, act_dim: int, device: str):
        self.act_dim, self.device = act_dim, device
        self.clear()

    def __len__(self): return int(self.n_steps)

    def clear(self):
        # Flattened storage for performance
        self.obs, self.act, self.logp, self.val, self.adv, self.ret, self.next_hp = [], [], [], [], [], [], []
        self.ep_ids = []  # New: tracks boundaries via flat IDs
        self.n_steps = 0
        self.ep_counter = 0 
        self._tensor_cache = None

    def add_steps(self, obs, act, logp, val, adv, ret, next_hp, ep_ids):
        """
        Maintains the flat plumbing. 'ep_ids' should match the length of the steps.
        """
        self._tensor_cache = None
        def dev(x): return x.detach().to(self.device)
        self.obs.append(obs.detach().to("cpu", non_blocking=True))
        self.act.append(dev(act)); self.logp.append(dev(logp))
        self.val.append(dev(val)); self.adv.append(dev(adv))
        self.ret.append(dev(ret)); self.next_hp.append(dev(next_hp))
        self.ep_ids.append(dev(ep_ids))
        self.n_steps += int(act.shape[0])

    def tensorize(self) -> Tuple:
        if self._tensor_cache: return self._tensor_cache
        self._tensor_cache = (
            torch.cat(self.obs, dim=0), torch.cat(self.act, dim=0), 
            torch.cat(self.logp, dim=0).float(), torch.cat(self.val, dim=0).float(), 
            torch.cat(self.adv, dim=0).float(), torch.cat(self.ret, dim=0).float(), 
            torch.cat(self.next_hp, dim=0).float(),
            torch.cat(self.ep_ids, dim=0) # Index 7: Episode Boundaries
        )
        return self._tensor_cache
    
    def swap_out_tensor_cache(self):
        data = self.tensorize(); self.clear(); return data
        
    def _episode_bounds(self, ep_ids: torch.Tensor) -> list[tuple[int, int]]:
        n = int(ep_ids.numel())
        if n == 0:
            return []

        changes = torch.nonzero(ep_ids[1:] != ep_ids[:-1], as_tuple=False).flatten() + 1
        bounds = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ep_ids.device),
            changes,
            torch.tensor([n], dtype=torch.long, device=ep_ids.device),
        ])
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.numel() - 1)]

    def iter_minibatches(self, mb_size: int, shuffle_episodes: bool = False) -> Iterator[Tuple]:
        # Every observation is independent in this stateless model, and every
        # state is trained on. A small remainder joins the last minibatch rather
        # than taking its own optimizer step (see minibatch_bounds).
        data = self.tensorize()
        n = len(data[1])
        order = torch.randperm(n) if shuffle_episodes else torch.arange(n)
        for start, end in minibatch_bounds(n, mb_size):
            idx = order[start:end]
            yield tuple(d[idx] for d in data)


def minibatch_bounds(n: int, size: int) -> List[Tuple[int, int]]:
    """Minibatch [start, end) ranges covering all n states.

    A remainder smaller than half a minibatch is merged into the last full
    minibatch: Adam takes equal-sized steps regardless of batch size, so a
    separate step on a handful of states would be mostly noise. The README-era
    code dropped remainders; merging keeps every state in the update.
    """
    bounds = [(start, min(start + size, n)) for start in range(0, n, size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < size // 2:
        tail = bounds.pop()
        bounds[-1] = (bounds[-1][0], tail[1])
    return bounds


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
                    policy_loss = F.cross_entropy(logits.float(), actions, label_smoothing=.1)
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
                        logp, entropies = masked_logprob_entropy(logits.float(), mask, actions)
                        log_ratio = logp - old_logp
                        ratio = log_ratio.exp()
                        coefficient = cfg.get("clip_coef", .2)
                        policy_loss = torch.maximum(-advantages*ratio,
                            -advantages*ratio.clamp(1-coefficient, 1+coefficient)).mean()
                        entropy = entropies.mean()
                        kl = ((ratio-1)-log_ratio).mean()
                        signed = -log_ratio.mean()
                        clip = ((ratio-1).abs() > coefficient).float().mean()
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
            check = batch_values["approx_kl"] if stop_rule == "minibatch" else epoch_signed_kl/epoch_batches
            eligible = stop_rule == "minibatch" or epoch > 0
            if mode.startswith("ppo") and cfg.get("target_kl") and eligible and check > 1.5*cfg["target_kl"]:
                stop = True
                stop_kl = check
                break
        if epoch_samples == len(dataset):
            epochs_completed += 1
        if stop:
            break
    return PPOUpdateStats(**{name: value/batches for name,value in sums.items()}, n_mb=batches,
        n_microbatches=microbatches, samples_processed=samples_processed,
        first_pass_samples=first_pass_samples, first_pass_fraction=first_pass_samples/len(dataset),
        epochs_completed=epochs_completed, early_stopped=stop, stop_kl=stop_kl)
