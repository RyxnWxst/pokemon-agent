"""FP32 Ray learner for imitation, critic warmup, and 15-token PPO."""
import asyncio
from dataclasses import asdict
import logging
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from checkpoint_io import load_weights, read_checkpoint, replace_file
from optimizers import make_optimizer
from ppo_core import AsyncEpisodeDataset, gae_from_episode, ppo_update

logger = logging.getLogger(__name__)


class LearnerActor:
    def __init__(self, cfg, weight_store):
        self.run_cfg, self.cfg, self.weight_store = cfg, cfg.learner, weight_store
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        torch.manual_seed(71)
        np.random.seed(71)
        random.seed(71)
        self.net = cfg.make_model().to(self.cfg.device)
        self.opt, self.sched = make_optimizer(self.net, self.cfg, progress=self.state_progress)
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

    def state_progress(self):
        """Fraction of this phase's state target reached (drives the linear schedule)."""
        target = self.cfg.max_total_steps
        return min(1., getattr(self, "total_steps", 0) / target) if target else 0.

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
        obs, actions, logp, values, variances, rewards, dones, lengths = batch
        if sum(lengths) != len(actions):
            raise ValueError("Packed episode lengths do not match action count")
        # Ray exposes read-only NumPy views. Own the tensors before training.
        obs = torch.tensor(obs)
        actions = torch.tensor(actions, dtype=torch.long)
        logp = torch.tensor(logp, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        variances = torch.tensor(variances, dtype=torch.float32)
        advantages, returns, ids = [], [], []
        start = 0
        for length in lengths:
            end = start + int(length)
            adv, ret = gae_from_episode(rewards[start:end], values[start:end], dones[start:end],
                self.cfg.gamma, self.cfg.gae_lambda,
                variances=variances[start:end] if self.cfg.use_dynamic_lambda else None,
                lam_min=self.cfg.dynamic_lambda_min, lam_max=self.cfg.dynamic_lambda_max,
                lam_eps=self.cfg.dynamic_lambda_eps)
            advantages.append(adv)
            returns.append(ret)
            ids.append(torch.full((int(length),), self.total_episodes, dtype=torch.long))
            self.total_episodes += 1
            start = end
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
        data[4] = ((data[4]-data[4].mean()) / (data[4].std(unbiased=False)+1e-8)).clamp(-10, 10)
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
            self._save_checkpoint(self.directory / f"learner_update_{self.update_idx:09d}.pt")
        self._maybe_save_milestone()

    def _maybe_save_milestone(self):
        """Keep a permanent policy-only copy each time new states cross a milestone.

        Regular checkpoints rotate (keep_last); milestones are never deleted and
        load directly in evaluate.py. A later run started from one gets a fresh optimizer.
        """
        every = self.cfg.milestone_every_states
        if not every:
            return None
        reached = self.total_steps // every
        if reached <= getattr(self, "last_milestone", 0):
            return None
        self.last_milestone = reached
        path = self.directory / "milestones" / f"states-{reached*every:012d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(project="ps-ppo-15token-milestone", architecture="readme_15token", format_version=1,
            phase=self.cfg.mode, model=self.net.state_dict(), model_config=self.net.reproduction_config,
            milestone_states=reached*every, total_steps=self.total_steps, update_idx=self.update_idx,
            total_episodes=self.total_episodes, run_cfg=self.run_cfg.as_dict())
        temporary = path.with_suffix(path.suffix+".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
        print(f"[learner] milestone {path.name} at new_states={self.total_steps} update={self.update_idx}", flush=True)
        return str(path.resolve())

    def _save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(project="ps-ppo-15token", architecture="readme_15token", format_version=1,
            model=self.net.state_dict(), model_config=self.net.reproduction_config,
            run_cfg=self.run_cfg.as_dict(), phase=self.cfg.mode, optimizer=self.opt.state_dict(),
            scheduler=self.sched.state_dict(), update_idx=self.update_idx, total_steps=self.total_steps,
            total_episodes=self.total_episodes, torch_rng=torch.get_rng_state(),
            training_totals=getattr(self, "training_totals", {}),
            cuda_rng=torch.cuda.get_rng_state_all() if self.cfg.device.startswith("cuda") else None,
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            pending=self.dataset.tensorize() if len(self.dataset) else None)
        temporary = path.with_suffix(path.suffix+".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
        if self.cfg.keep_last > 0:
            for old in sorted(self.directory.glob("learner_update_*.pt"))[:-self.cfg.keep_last]:
                old.unlink()
        return str(path.resolve())

    def _load_checkpoint(self, path):
        checkpoint = read_checkpoint(path)
        load_weights(self.net, checkpoint)
        same_phase = checkpoint.get("project") == "ps-ppo-15token" and checkpoint.get("phase") == self.cfg.mode
        if same_phase:
            self._check_optimizer_settings(checkpoint)
            self.opt.load_state_dict(checkpoint["optimizer"])
            self.sched.load_state_dict(checkpoint["scheduler"])
            self.update_idx = checkpoint["update_idx"]
            self.total_steps = checkpoint["total_steps"]
            self.total_episodes = checkpoint["total_episodes"]
            self.training_totals = {**getattr(self, "training_totals", {}), **checkpoint.get("training_totals", {})}
            if checkpoint.get("pending") is not None:
                self.dataset.add_steps(*checkpoint["pending"])
            torch.set_rng_state(checkpoint["torch_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            random.setstate(checkpoint["python_rng"])
            if self.cfg.device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"[learner] loaded {path}; "
              f"{'resumed optimizer/counters' if same_phase else 'new phase with pretrained weights'}", flush=True)

    def _check_optimizer_settings(self, checkpoint):
        """Same-phase resume restores saved rates; refuse settings that would be silently ignored."""
        saved = checkpoint.get("run_cfg", {}).get("learner", {})
        # Checkpoints written before these settings existed used these values.
        saved = {"lr_schedule": "historical", "ppo_lr_multipliers": "current", **saved}
        names = ["lr", "lr_schedule"] + (["ppo_lr_multipliers"] if self.cfg.mode.startswith("ppo") else [])
        changed = {name: (saved.get(name), getattr(self.cfg, name)) for name in names
            if saved.get(name) != getattr(self.cfg, name)}
        if changed:
            raise ValueError(f"Resuming {self.cfg.mode} would keep the checkpoint's optimizer settings, "
                f"not the requested ones (saved, requested): {changed}. Pass the saved settings to "
                "resume this run, or start a new run from an imitation or warmup checkpoint.")

    async def save_now(self, path=None):
        return self._save_checkpoint(path or self.directory / f"learner_update_{self.update_idx:09d}.pt")

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
