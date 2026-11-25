"""Batched, stateless policy inference shared by both self-play sides."""
import numpy as np
import ray
import torch
import time
from pokemon_agent.model.transformer import make_model
from pokemon_agent.rl.distributions import masked_sample
from pokemon_agent.rl.value import value_variance


class InferenceActor:
    def __init__(self, cfg, weight_store):
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        self.cfg, self.weight_store = cfg, weight_store
        self.net = make_model(cfg.model).to(cfg.infer.device).eval()
        self.net.requires_grad_(False)
        self.current_version = -1
        self.requests = self.flushes = 0
        self.started = time.perf_counter()
        self.started_cpu = time.process_time()
        self.sync_seconds = self.inference_seconds = 0.

    def _sync_weights(self):
        """Load the newest published weights if the store's version has moved on."""
        version = ray.get(self.weight_store.get_version.remote())
        if version != self.current_version:
            weights = ray.get(self.weight_store.get_weights.remote())
            if weights is None:
                raise RuntimeError("Learner has not published initial weights")
            self.net.load_state_dict(weights, strict=True)
            self.current_version = version

    @torch.no_grad()
    def infer_batch(self, tags, observations):
        if len(tags) != len(observations):
            raise ValueError("Battle tags and observations must have the same length")
        started = time.perf_counter()
        self._sync_weights()
        self.sync_seconds += time.perf_counter()-started
        started = time.perf_counter()
        obs = torch.tensor(np.asarray(observations), device=self.cfg.infer.device)
        logits, value_logits, values, *_ = self.net(obs)
        variance = value_variance(value_logits, values, self.net.v_support)
        start, end = self.net.unpacker.offsets["action_mask"]
        actions, logp, _ = masked_sample(logits, obs[:, start:end], temp=1.)
        self.requests += len(obs)
        self.flushes += 1
        result = tuple(t.cpu().numpy() for t in (actions, logp, values, variance))
        self.inference_seconds += time.perf_counter()-started
        return result

    def clear_cache(self, tag):
        # The rollout bridge sends this notification for all model types.
        # A stateless 15-token policy has no per-battle inference cache.
        return None

    def get_stats(self):
        return dict(total_requests=self.requests, model_version=self.current_version,
            avg_batch_size=self.requests/max(1, self.flushes), tokens=15, recurrent=False,
            weight_sync_seconds=self.sync_seconds, inference_seconds=self.inference_seconds,
            process_cpu_cores_average=(time.process_time()-self.started_cpu)/max(.001, time.perf_counter()-self.started),
            gpu_allocated_bytes=torch.cuda.memory_allocated() if self.cfg.infer.device.startswith("cuda") else 0,
            gpu_reserved_bytes=torch.cuda.memory_reserved() if self.cfg.infer.device.startswith("cuda") else 0)
