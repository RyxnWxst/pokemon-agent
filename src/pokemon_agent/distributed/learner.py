"""FP32 Ray learner for imitation, critic warmup, and 15-token PPO."""
import asyncio
from dataclasses import asdict
import logging
from pathlib import Path
import random
import time
import numpy as np
import torch

from pokemon_agent.model.transformer import make_model
from pokemon_agent.rl.advantages import episode_advantages, normalize_advantages
from pokemon_agent.rl.buffer import AsyncEpisodeDataset, unpack_episodes
from pokemon_agent.rl.ppo import ppo_update
from pokemon_agent.training.checkpoints import (learner_checkpoint_name, restore_learner_checkpoint,
    save_learner_checkpoint, save_milestone_if_due)
from pokemon_agent.training.optimizer import make_optimizer

logger = logging.getLogger(__name__)


class LearnerActor:
    def __init__(self, cfg, weight_store):
        self.run_cfg, self.cfg, self.weight_store = cfg, cfg.learner, weight_store
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        torch.manual_seed(71)
        np.random.seed(71)
        random.seed(71)
        self.net = make_model(cfg.model).to(self.cfg.device)
        self.opt, self.sched = make_optimizer(self.net, self.cfg)
        self.dataset = AsyncEpisodeDataset(cfg.env.act_dim, "cpu")
        self.update_idx = self.total_steps = self.total_episodes = 0
        self.last_update = None
        self.training_totals = dict(updates=0, offered_states=0, first_pass_samples=0,
            samples_processed=0, optimizer_steps=0, early_stops=0)
        self.started = time.perf_counter()
        self.started_cpu = time.process_time()
        self.directory = Path(self.cfg.ckpt_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.cfg.resume:
            paths = sorted(self.directory.glob("learner_update_*.pt"))
            path = paths[-1] if paths else self.cfg.resume_path
            if path:
                self._load_checkpoint(path)
        # Milestones already passed (including before a resume) are not re-saved.
        every = self.cfg.milestone_every_states
        self.last_milestone = self.total_steps // every if every else 0
        self._q = asyncio.Queue(maxsize=cfg.rollout.learn_max_pending_batches)
        loop = asyncio.get_event_loop()
        self._startup_task = loop.create_task(self._publish())
        self._task = loop.create_task(self._loop())
        self._task.add_done_callback(self._on_loop_done)

    async def _publish(self):
        weights = {key: value.detach().cpu() for key, value in self.net.state_dict().items()}
        await self.weight_store.update.remote(weights, self.update_idx)

    def _on_loop_done(self, task):
        if not task.cancelled() and task.exception() is not None:
            logger.error("Learner failed", exc_info=task.exception())

    async def ready(self):
        await self._startup_task
        return True

    async def submit_packed_batch(self, *batch):
        await self._q.put(batch)
        return True

    def _consume(self, batch):
        obs, actions, logp, values, variances, rewards, dones, lengths = unpack_episodes(batch)
        advantages, returns = episode_advantages(rewards, values, dones, variances, lengths, self.cfg)
        ids = []
        for length in lengths:
            ids.append(torch.full((length,), self.total_episodes, dtype=torch.long))
            self.total_episodes += 1
        self.dataset.add_steps(obs, actions, logp, values, torch.cat(advantages), torch.cat(returns),
            torch.zeros(len(actions)), torch.cat(ids))
        # Only newly received player decisions count; optimizer passes never do.
        self.total_steps += len(actions)

    async def _loop(self):
        await self._startup_task
        while True:
            if self.cfg.max_updates and self.update_idx >= self.cfg.max_updates:
                return
            if self.cfg.max_total_steps and self.total_steps >= self.cfg.max_total_steps:
                if len(self.dataset):
                    await self._update()
                return
            batches = [await self._q.get()]
            while not self._q.empty():
                batches.append(self._q.get_nowait())
            for batch in batches:
                self._consume(batch)
            reached_target = self.cfg.max_total_steps and self.total_steps >= self.cfg.max_total_steps
            if len(self.dataset) >= self.cfg.steps_per_update or reached_target:
                await self._update()

    async def _update(self):
        data = list(self.dataset.swap_out_tensor_cache())
        data[4] = normalize_advantages(data[4])
        training = AsyncEpisodeDataset(self.run_cfg.env.act_dim, "cpu")
        training.add_steps(*data)
        stats = ppo_update(self.net, self.opt, training, scheduler=self.sched, mode=self.cfg.mode,
            **self.cfg.ppo_kwargs(), v_min=self.cfg.v_min, v_max=self.cfg.v_max, v_bins=self.cfg.v_bins)
        self.update_idx += 1
        self.last_update = asdict(stats)
        self.training_totals["updates"] += 1
        self.training_totals["offered_states"] += len(training)
        self.training_totals["first_pass_samples"] += stats.first_pass_samples
        self.training_totals["samples_processed"] += stats.samples_processed
        self.training_totals["optimizer_steps"] += stats.n_mb
        self.training_totals["early_stops"] += int(stats.early_stopped)
        await self._publish()
        print(f"[learner] update={self.update_idx} new_states={self.total_steps} "
              f"loss={stats.total_loss:.4f} kl={stats.approx_kl:.5f} "
              f"first_pass={stats.first_pass_fraction:.3f} optimizer_steps={stats.n_mb}", flush=True)
        if self.update_idx % self.cfg.save_every_updates == 0:
            self._save_checkpoint(self.directory / learner_checkpoint_name(self.update_idx))
        self._maybe_save_milestone()

    def _maybe_save_milestone(self):
        self.last_milestone, path = save_milestone_if_due(self.directory, self.net, self.run_cfg,
            last_milestone=getattr(self, "last_milestone", 0), total_steps=self.total_steps,
            update_idx=self.update_idx, total_episodes=self.total_episodes)
        if path:
            print(f"[learner] milestone {Path(path).name} at new_states={self.total_steps} "
                  f"update={self.update_idx}", flush=True)
        return path

    def _save_checkpoint(self, path):
        return save_learner_checkpoint(path, self.net, self.opt, self.sched, self.run_cfg, self.dataset,
            update_idx=self.update_idx, total_steps=self.total_steps, total_episodes=self.total_episodes,
            training_totals=getattr(self, "training_totals", {}))

    def _load_checkpoint(self, path):
        restored = restore_learner_checkpoint(path, self.net, self.opt, self.sched, self.dataset, self.cfg)
        if restored:
            self.update_idx = restored["update_idx"]
            self.total_steps = restored["total_steps"]
            self.total_episodes = restored["total_episodes"]
            self.training_totals = {**getattr(self, "training_totals", {}), **restored["training_totals"]}
        print(f"[learner] loaded {path}; "
              f"{'resumed optimizer/counters' if restored else 'new phase with pretrained weights'}", flush=True)

    async def save_now(self, path=None):
        return self._save_checkpoint(path or self.directory / learner_checkpoint_name(self.update_idx))

    async def get_stats(self):
        for task in (self._startup_task, self._task):
            if task.done():
                task.result()
        return dict(update=self.update_idx, total_steps=self.total_steps, episodes=self.total_episodes,
            steps_in_dataset=len(self.dataset), completed=self._task.done(), last_update=self.last_update,
            training_totals=self.training_totals,
            process_cpu_cores_average=(time.process_time()-self.started_cpu)/max(.001, time.perf_counter()-self.started),
            gpu_allocated_bytes=torch.cuda.memory_allocated() if self.cfg.device.startswith("cuda") else 0,
            gpu_reserved_bytes=torch.cuda.memory_reserved() if self.cfg.device.startswith("cuda") else 0)
