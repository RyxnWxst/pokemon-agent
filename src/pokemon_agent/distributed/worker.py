"""
Distributed Rollout Worker for Pokémon Showdown RL.

This module uses a Thread-Bridge architecture to quarantine poke-env's async
websocket logic into a background thread (env/bridge.py). The main thread operates
completely synchronously, gathering perfect numpy batches and dispatching them to a
central GPU Inference Actor via zero-copy Ray RPCs.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
import ray

from pokemon_agent.config import RunConfig
from pokemon_agent.distributed.learner_client import SyncLearnerClient
from pokemon_agent.env.bridge import PokeEnvVectorWrapper
from pokemon_agent.env.patches import apply_all
from pokemon_agent.obs.assembler import ObservationAssembler
from pokemon_agent.rl.rewards import trajectory_rewards

logger = logging.getLogger(__name__)


class RolloutWorker:
    def __init__(self, cfg: RunConfig, inference_actor, learner_actor, pairs: int, server_port: int):
        # Each Ray actor is its own process, so it applies the poke-env patches itself.
        apply_all()
        self.cfg = cfg
        self.started = time.perf_counter()
        self.started_cpu = time.process_time()
        self.learner_client = SyncLearnerClient(learner_actor, cfg)

        # The central inference actor that picks actions for every battle in this worker
        self.inference_actor = inference_actor

        self.vec_env = PokeEnvVectorWrapper(cfg, pairs, server_port)
        self._traj = {}
        self.model_games = 0
        self.model_wins = 0
        self.assembler = ObservationAssembler()

    async def run(self):
        rollout_thread = threading.Thread(target=self._run_rl_loop, daemon=True)
        rollout_thread.start()
        while True:
            await asyncio.sleep(1.0)
            if not self.learner_client.thread.is_alive():
                raise RuntimeError(self.learner_client.failure or "Learner submission thread stopped")
            if not rollout_thread.is_alive():
                raise RuntimeError("Rollout thread stopped; inspect the preceding exception")
            if not self.vec_env.bg_thread.is_alive():
                raise RuntimeError("Battle connection thread stopped; inspect the preceding exception")

    def _run_rl_loop(self):
        actions_dict = {}

        while True:
            events = self.vec_env.step(actions_dict)
            actions_dict = {}

            step_tags = []
            step_obs = []

            for event_type, tag, payload, extra_data in events:
                if event_type == "ERROR":
                    raise RuntimeError(f"Battle processing failed for {tag}: {payload}")
                elif event_type == "STEP":
                    step_tags.append(tag)
                    step_obs.append(payload)
                elif event_type == "IMITATION":
                    # payload is the observation, extra_data the teacher's action index
                    self._record_step(tag, payload, extra_data, 0.0, 0.0, 0.0)
                elif event_type == "DONE":
                    self._finalize_trajectory(tag, extra_data)
                elif event_type == "CLEANUP":
                    self._cleanup_trajectory(tag)

            if step_obs:
                obs_batch = np.ascontiguousarray(np.stack(step_obs, axis=0), dtype=np.float32)

                # --- FIRE TO CENTRAL INFERENCE GPU ---
                # This blocks the worker thread until Ray returns the answers.
                # Because obs_batch is contiguous np.float32, Ray zero-copies it!
                try:
                    acts, logps, vals, var = ray.get(
                        self.inference_actor.infer_batch.remote(step_tags, obs_batch)
                    )
                except Exception as e:
                    logger.error(f"Inference Actor call failed: {e}")
                    raise

                for i, tag in enumerate(step_tags):
                    actions_dict[tag] = int(acts[i])
                    self._record_step(tag, step_obs[i], acts[i], logps[i], vals[i], var[i])

    def _record_step(self, tag: str, obs, action, logp, value, variance):
        """Append one decision to this player-battle's trajectory."""
        traj = self._traj.setdefault(tag, {"obs": [], "act": [], "logp": [], "val": [], "var": []})
        traj["obs"].append(obs)
        traj["act"].append(action)
        traj["logp"].append(logp)
        traj["val"].append(value)
        traj["var"].append(variance)

    def _cleanup_trajectory(self, tag: str):
        """Called when a Zombie/Stall is forcibly killed by the async maintenance thread."""
        self.inference_actor.clear_cache.remote(tag) # Fire and forget
        self._traj.pop(tag, None)
        self.vec_env.release_slot()

    def _finalize_trajectory(self, tag: str, battle):
        """Calculates rewards and sends to learner."""
        self.inference_actor.clear_cache.remote(tag) # Fire and forget

        if tag in self._traj:
            buf = self._traj.pop(tag)
            if buf["act"]:
                if self.cfg.learner.mode == "ppo_vs_teacher":
                    self.model_games += 1
                    self.model_wins += int(bool(battle.won))
                obs_stacked = np.stack(buf["obs"], axis=0)
                rewards, dones = trajectory_rewards(obs_stacked, battle.won, battle.lost,
                    self.cfg.reward, self.assembler.meta)

                self.learner_client.submit_episode(
                    obs_stacked, np.array(buf["act"]), np.array(buf["logp"]),
                    np.array(buf["val"]), np.array(buf["var"]), rewards, dones
                )

        self.vec_env.release_slot()

    def heartbeat(self):
        """Called by Train.py telemetry."""
        sem_val = getattr(self.vec_env, "ep_sem", None)
        return {
            "process_cpu_cores_average": (time.process_time()-self.started_cpu)/max(.001, time.perf_counter()-self.started),
            "active_battles_worker": len({tag for pA, pB in self.vec_env.active_pairs for p in (pA, pB) for tag in getattr(p, "_battles", {})}),
            # Should stay near the number of in-flight messages; growth means stuck handlers.
            "pending_message_tasks": sum(p.pending_message_tasks() for pA, pB in self.vec_env.active_pairs for p in (pA, pB)),
            "finished_rooms_tracked": sum(len(p._finished_rooms) for pA, pB in self.vec_env.active_pairs for p in (pA, pB)),
            "learner_q_size": self.learner_client.q.qsize(),
            "traj_in_memory": len(self._traj),
            "ep_sem_value": sem_val._value if sem_val else -1,
            "model_games": self.model_games,
            "model_wins": self.model_wins,
        }
