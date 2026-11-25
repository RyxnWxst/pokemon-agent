"""The rollout dataset, minibatching, and packing episodes to ship between processes."""
from __future__ import annotations
from typing import Iterator, List, Tuple
import numpy as np
import torch


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


def make_dataset(episodes):
    """Dataset from (observations, actions, returns) episodes, as imitation shards store them."""
    dataset = AsyncEpisodeDataset(14, "cpu")
    for index, (obs, actions, returns) in enumerate(episodes):
        zeros = torch.zeros(len(actions))
        dataset.add_steps(obs, actions, zeros, zeros, zeros, returns, zeros,
            torch.full((len(actions),), index, dtype=torch.long))
    return dataset


def pack_episodes(items: List[Tuple]) -> Tuple:
    """Concatenate finished episodes (obs, act, logp, val, var, rew, done) into one batch plus their lengths."""
    lengths = np.asarray([it[1].shape[0] for it in items], dtype=np.int32)
    return (
        np.concatenate([it[0] for it in items], axis=0),
        np.concatenate([it[1] for it in items], axis=0).astype(np.int64),
        np.concatenate([it[2] for it in items], axis=0).astype(np.float32),
        np.concatenate([it[3] for it in items], axis=0).astype(np.float32),
        np.concatenate([it[4] for it in items], axis=0).astype(np.float32), # var_cat
        np.concatenate([it[5] for it in items], axis=0).astype(np.float32), # rew_cat
        np.concatenate([it[6] for it in items], axis=0).astype(np.float32), # done_cat
        lengths
    )


def unpack_episodes(batch) -> Tuple:
    """Check a packed batch and own its arrays as tensors (Ray hands over read-only NumPy views)."""
    obs, actions, logp, values, variances, rewards, dones, lengths = batch
    if sum(lengths) != len(actions):
        raise ValueError("Packed episode lengths do not match action count")
    return (torch.tensor(obs), torch.tensor(actions, dtype=torch.long),
        torch.tensor(logp, dtype=torch.float32), torch.tensor(values, dtype=torch.float32),
        torch.tensor(variances, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(dones, dtype=torch.float32), [int(length) for length in lengths])
