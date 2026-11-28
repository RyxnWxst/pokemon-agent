"""
Distributed Rollout Worker for Pokémon Showdown RL.

This module uses a Thread-Bridge architecture to quarantine poke-env's async 
websocket logic into a background thread. The main thread operates completely 
synchronously, gathering perfect numpy batches and dispatching them to a central 
GPU Inference Actor via zero-copy Ray RPCs.
"""

from __future__ import annotations

import random
import asyncio
import functools
import logging
import sys
import time
import traceback
import threading
import queue
import secrets
from typing import Any, Dict, List, Tuple, Final
from collections import Counter

import numpy as np
import ray
import poke_env
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import DefaultBattleOrder
from poke_env.player.baselines import SimpleHeuristicsPlayer
from config import RunConfig
from obs_assembler import ObservationAssembler
from rewards import episode_rewards

# Global Constants
ZOMBIE_LIMIT_S: Final[float] = 300.0  
REPAIR_INTERVAL_S: Final[float] = 30.0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MONKEY PATCHES & SYSTEM FIXES
# ---------------------------------------------------------------------------

import math

def _corrected_stat_estimation(mon, stat: str) -> float:
    """
    A mathematically accurate stat estimator for Gen 9 Random Battles.
    Replaces the hallucinated Level 100 / 0 EV logic in the original bot.
    """
    # 1. FIX THE BOOST MULTIPLIER
    # Original bug checked `> 1` which made +1 act like +0, and +2 act like +2.
    boost_lvl = mon.boosts.get(stat, 0)
    if boost_lvl > 0:
        boost_mult = (2 + boost_lvl) / 2  # e.g., +1 = 1.5x
    else:
        boost_mult = 2 / (2 - boost_lvl)  # e.g., -1 = 0.66x

    # 2. APPLY TRUE GEN 9 RANDBATS MATH
    base = mon.base_stats[stat]
    level = mon.level
    
    # Gen 9 RandBats assigns a flat 85 EVs to every stat
    ev = 85 
    
    # We assume 31 IVs (ignoring the 0 Speed Trick Room edge case for simplicity)
    iv = 31 
    
    # Standard Pokémon non-HP stat formula
    raw_stat = math.floor(((2 * base + iv + math.floor(ev / 4)) * level) / 100) + 5
    
    # Apply boosts
    return raw_stat * boost_mult

# Apply the Monkey Patch
SimpleHeuristicsPlayer._stat_estimation = _corrected_stat_estimation

_original_handle_message = poke_env.ps_client.PSClient._handle_message

def _apply_patches():
    """Applies necessary patches to poke_env and asyncio for stability."""
    original_handle = poke_env.ps_client.PSClient._handle_message
    async def loud_handle(self, message):
        try:
            if "|error|[Invalid choice]" in message:
                room_id = message.split("\n")[0].replace(">", "").strip()
                logger.debug(f"NN chose invalid move in {room_id}. Forcing /choose default.")
                await self.send_message("/choose default", room=room_id)
            await original_handle(self, message)
        except Exception:
            logger.critical(f"Poke-env crash on message: {message[:100]}...")
            traceback.print_exc()
            raise
    poke_env.ps_client.PSClient._handle_message = loud_handle

    if sys.platform == "win32":
        import asyncio.proactor_events
        def silence_proactor_error(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                try: return func(self, *args, **kwargs)
                except (AssertionError, OSError): pass
            return wrapper
        asyncio.proactor_events._ProactorBaseWritePipeTransport._loop_writing = \
            silence_proactor_error(asyncio.proactor_events._ProactorBaseWritePipeTransport._loop_writing)

_apply_patches()

def make_server_conf(host: str, port: int) -> ServerConfiguration:
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    http_action = f"http://{host}:{port}/action.php?"
    return ServerConfiguration(ws_url, http_action)

def battle_tag_for(battle) -> str:
    tag = getattr(battle, "battle_tag", None)
    if not tag: tag = f"pyid_{id(battle)}"
    return str(tag)

def mk_name(run_tag: str, p: int, side: str) -> str:
    return f"p{run_tag}{p:03d}{side}"

# --- RESTORED NETWORKING HELPERS ---
async def wait_for_login(player: SyncBridgePlayer, timeout_s: float = 30.0) -> None:
    c = getattr(player, "ps_client", None)
    if c is None: raise RuntimeError(f"{player.username} has no ps_client")
    fn = getattr(c, "wait_for_login", None)
    if not callable(fn): raise RuntimeError(f"{player.username}.ps_client has no wait_for_login()")
    await asyncio.wait_for(fn(), timeout=timeout_s)

async def safe_send(player: SyncBridgePlayer, message: str, room: str, *, retries: int = 8) -> None:
    await wait_for_login(player)
    last_err = None
    for i in range(retries):
        try:
            await player.ps_client.send_message(message, room=room)
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.05 * (i + 1))
    raise RuntimeError(f"[safe_send] failed: {message!r} ({last_err!r})")

async def join_lobby(player: SyncBridgePlayer) -> None:
    await safe_send(player, "/join lobby", room="")

# ---------------------------------------------------------------------------
# SYNC LEARNER CLIENT
# ---------------------------------------------------------------------------

class SyncLearnerClient:
    def __init__(self, learner_actor, cfg: RunConfig):
        self.learner_actor = learner_actor
        self.cfg = cfg.rollout
        self.q = queue.Queue(maxsize=self.cfg.learn_queue_episodes
            or self.cfg.learn_max_pending_batches * self.cfg.learn_max_episodes)
        self.failure = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit_episode(self, obs: np.ndarray, act: np.ndarray, logp: np.ndarray, 
                       val: np.ndarray, var: np.ndarray, rew: np.ndarray, done: np.ndarray) -> bool:
        # Apply backpressure rather than silently discarding completed games.
        if self.failure:
            raise RuntimeError(self.failure)
        self.q.put((obs, act, logp, val, var, rew, done))
        return True

    def _worker(self):
        while True:
            items = [self.q.get()]
            while len(items) < self.cfg.learn_max_episodes:
                try: items.append(self.q.get_nowait())
                except queue.Empty: break
            
            packed = self._prepare_batch(items)
            try:
                ray.get(self.learner_actor.submit_packed_batch.remote(*packed))
            except Exception as e:
                self.failure = f"Learner submission failed: {e!r}"
                logger.error(self.failure)
                return

    @staticmethod
    def _prepare_batch(items: List[Tuple]) -> Tuple:
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

# ---------------------------------------------------------------------------
# TOLLBOOTH PLAYER (RUNS IN ASYNC THREAD)
# ---------------------------------------------------------------------------

FINISHED_ROOM_TTL_S: Final[float] = 600.0


class FinishedRoomFilter:
    """Drop messages for battle rooms this player has finished and forgotten.

    Rollout players remove finished battles from poke-env's table to bound memory.
    poke-env then leaves the room, and Showdown answers with a final |deinit for it.
    For a battle missing from its table, poke-env's _get_battle waits forever, and
    every later battle start wakes all such waiters, so per-decision CPU grew with
    the number of battles a worker had finished. Those messages are now ignored.
    """
    def _forget_room(self, room_id: str) -> None:
        self._battles.pop(room_id, None)
        self._finished_rooms.pop(room_id, None)
        now = time.monotonic()
        self._finished_rooms[room_id] = now
        # Insertion order is time order: prune expired rooms whose |deinit never came.
        while self._finished_rooms:
            oldest, stamp = next(iter(self._finished_rooms.items()))
            if now - stamp < FINISHED_ROOM_TTL_S:
                break
            del self._finished_rooms[oldest]

    def _is_finished_room(self, split_messages) -> bool:
        if not split_messages or not split_messages[0] or not split_messages[0][0].startswith(">"):
            return False
        room_id = split_messages[0][0][1:].strip()
        if room_id not in self._finished_rooms:
            return False
        if any(len(parts) > 1 and parts[1] == "deinit" for parts in split_messages[1:]):
            del self._finished_rooms[room_id]  # Showdown sends nothing further for this room.
        return True

    def pending_message_tasks(self) -> int:
        return len(getattr(self.ps_client, "_active_tasks", ()))


class TrainingHeuristicPlayer(FinishedRoomFilter, SimpleHeuristicsPlayer):
    def __init__(self, *args, **kwargs):
        self._finished_rooms = {}
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        if self._is_finished_room(split_messages):
            return
        await super()._handle_battle_message(split_messages)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self._forget_room(battle.battle_tag)


class SyncBridgePlayer(FinishedRoomFilter, SimpleHeuristicsPlayer):
    def __init__(self, cfg: RunConfig, event_queue: queue.Queue, action_futures: dict, ep_sem: asyncio.Semaphore, **kwargs):
        self._finished_rooms = {}
        super().__init__(**kwargs)
        self.cfg = cfg
        self.event_queue = event_queue
        self.action_futures = action_futures
        self.ep_sem = ep_sem
        self.assembler = ObservationAssembler()
        self._battle_starts = {}
        self._last_act_time = {}
        self._episode_slot_state = {} 
        self._battle_events = {}

    def _get_unique_tag(self, battle):
        return f"{self.username}_{battle_tag_for(battle)}"
    
    async def _handle_battle_message(self, split_messages: list[list[str]]):
        """Corrected for 0.15: Processes a batch of pre-split protocol lines."""
        if self._is_finished_room(split_messages):
            return
        if not split_messages or not split_messages[0]:
            await super()._handle_battle_message(split_messages)
            return

        # 1. Extract Room ID from the first line
        room_id = None
        if split_messages[0] and split_messages[0][0].startswith(">"):
            room_id = split_messages[0][0][1:].strip()

        # 2. Harvest Events
        if room_id:
            for parts in split_messages:
                # Valid protocol lines start with an empty string (from the '|' split)
                if len(parts) > 1 and parts[0] == "":
                    cmd = parts[1]
                    # Filter out structural tags we don't need for transitions
                    if cmd not in ("t:", "request", "turn"):
                        # Save the parts exactly as obs_transitions expects them
                        self._battle_events.setdefault(room_id, []).append(parts)

        # 3. Pass to parent so poke-env can update its internal Battle object
        await super()._handle_battle_message(split_messages)

    async def choose_move(self, battle):
        try:
            tag = self._get_unique_tag(battle)
            if getattr(battle, 'finished', False):
                return DefaultBattleOrder()
                
            now = time.time()
            self._last_act_time[tag] = now
            self._battle_starts.setdefault(tag, now)

            if tag not in self._episode_slot_state:
                await self.ep_sem.acquire()
                if tag not in self._battle_starts:
                    self._safe_release()
                    return DefaultBattleOrder()
                self._episode_slot_state[tag] = "acquired"
            
            events = self._battle_events.pop(battle.battle_tag, [])
            obs_flat = self.assembler.assemble(battle, events=events)
            mode = getattr(self.cfg.learner, "mode", "ppo")
            if mode in ("imitation", "imitation_frozen_backbone", "jepa_pretraining", "imitation_with_jepa"):
                if mode == "jepa_pretraining" and random.random() < 0.10:
                    order = self.choose_random_move(battle)
                else:
                    order = super().choose_move(battle)
                action_idx = self.assembler.map_order_to_index(order, battle)
                start, end = self.assembler.offsets["action_mask"]
                if obs_flat[start:end][action_idx] < 0.5:
                    raise RuntimeError("Teacher action is outside the legal action mask")
                
                # Send directly to the trajectory compiler
                self.event_queue.put_nowait(("IMITATION", tag, obs_flat, action_idx))
                
                action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
                if action_obj == "DEFAULT": return DefaultBattleOrder()
                return self.create_order(action_obj, **kwargs)

            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self.action_futures[tag] = (fut, loop) 

            if mode == "imitation_dagger":
                teacher_order = super().choose_move(battle)
                teacher_action = self.assembler.map_order_to_index(teacher_order, battle)
                start, end = self.assembler.offsets["action_mask"]
                if obs_flat[start:end][teacher_action] < 0.5:
                    raise RuntimeError("Teacher action is outside the legal action mask")
                self.event_queue.put_nowait(("DAGGER", tag, obs_flat, teacher_action))
            else:
                self.event_queue.put_nowait(("STEP", tag, obs_flat, battle))

            action_idx = await fut

            action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
            if action_obj == "DEFAULT": return DefaultBattleOrder()
            return self.create_order(action_obj, **kwargs)

        except Exception as e:
            logger.error(f"SyncBridgePlayer crashed for {battle.battle_tag}: {e}")
            self.event_queue.put_nowait(("ERROR", tag if 'tag' in locals() else battle.battle_tag, str(e), None))
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        tag = self._get_unique_tag(battle)
        self.event_queue.put_nowait(("DONE", tag, None, battle))
        
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)
        
        self.action_futures.pop(tag, None)
        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        self._forget_room(base_tag)
        self._battle_events.pop(base_tag, None)

        super()._battle_finished_callback(battle)

    def _cleanup_local_battle(self, tag: str):
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)
        
        self.action_futures.pop(tag, None) 
        
        self.event_queue.put_nowait(("CLEANUP", tag, None, None))

        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        self._forget_room(base_tag)
        self._battle_events.pop(base_tag, None)

    def _safe_release(self):
        try: self.ep_sem.release()
        except ValueError: pass

    async def run_reconciliation(self):
        try: await self.ps_client.send_message("/rlactive", room="")
        except Exception: pass

    def _handle_query(self, query_type: str, data: Any) -> None:
        super()._handle_query(query_type, data)
        if query_type == "rlactive":
            if not data or isinstance(data, list):
                server_ids = set()
            else:
                server_ids = set(str(data).split(","))
            
            server_ids.discard("")
            local_map = {tag: (tag.split("_", 1)[1] if "_" in tag else tag) for tag in self._battle_starts.keys()}
            local_base_ids = set(local_map.values())
            
            ghosts = list(server_ids - local_base_ids)
            for bid in ghosts:
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))
            
            now = time.time()
            stalls = [tag for tag, bid in local_map.items() if bid in server_ids and (now - self._last_act_time.get(tag, 0)) > 60.0]
            for tag in stalls:
                bid = local_map[tag]
                self._cleanup_local_battle(tag)
                asyncio.create_task(self.ps_client.send_message(f"/rlrescue {bid}", room=""))
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))
                
            zombies = [tag for tag, bid in local_map.items() if bid not in server_ids and (now - self._battle_starts.get(tag, 0)) > ZOMBIE_LIMIT_S]
            for tag in zombies:
                self._cleanup_local_battle(tag)

# ---------------------------------------------------------------------------
# VECTOR WRAPPER (THE BRIDGE)
# ---------------------------------------------------------------------------

class PokeEnvVectorWrapper:
    def __init__(self, cfg: RunConfig, pairs_count: int, server_port: int):
        self.cfg = cfg
        self.pairs_count = pairs_count
        self.server_conf = make_server_conf("127.0.0.1", server_port)
        self.run_tag = secrets.token_hex(3)
        
        self.event_queue = queue.Queue()
        self.action_futures = {} 
        self.active_pairs = []

        self.async_loop = asyncio.new_event_loop()
        self.bg_thread = threading.Thread(target=self._run_bg_loop, daemon=True)
        self.bg_thread.start()

    def _run_bg_loop(self):
        asyncio.set_event_loop(self.async_loop)
        self.ep_sem = asyncio.Semaphore(self.cfg.rollout.learn_max_pending_episodes)
        self.async_loop.run_until_complete(self._init_players_and_spawn())
        self.async_loop.run_forever()

    async def _init_players_and_spawn(self):
        for i in range(self.pairs_count):
            pA = SyncBridgePlayer(
                cfg=self.cfg,
                event_queue=self.event_queue, action_futures=self.action_futures, ep_sem=self.ep_sem,
                account_configuration=AccountConfiguration(mk_name(self.run_tag, i, "a"), None),
                server_configuration=self.server_conf, max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
                battle_format=self.cfg.env.battle_format, start_listening=True, log_level=40
            )
            opponent_args = dict(
                account_configuration=AccountConfiguration(mk_name(self.run_tag, i, "b"), None),
                server_configuration=self.server_conf, max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
                battle_format=self.cfg.env.battle_format, start_listening=True, log_level=40
            )
            if self.cfg.learner.heuristic_opponent:
                pB = TrainingHeuristicPlayer(**opponent_args)
            else:
                pB = SyncBridgePlayer(cfg=self.cfg, event_queue=self.event_queue,
                    action_futures=self.action_futures, ep_sem=self.ep_sem, **opponent_args)
            self.active_pairs.append((pA, pB))

        players = [p for pair in self.active_pairs for p in pair]
        
        async def wait_for_login_retry(player, attempts=5, delay=1.0):
            last_err = None
            for i in range(attempts):
                try:
                    await wait_for_login(player)
                    return
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(delay)
            raise last_err

        for p in players:
            await wait_for_login_retry(p)
            await asyncio.sleep(0.05)
            
        await asyncio.gather(*[join_lobby(p) for p in players])
        
        for (pA, pB) in self.active_pairs:
            await safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")

        self.async_loop.create_task(self._maintenance_loop())

    async def _maintenance_loop(self):
        loop_idx = 0
        while True:
            await asyncio.sleep(REPAIR_INTERVAL_S)
            loop_idx += 1
            for pA, pB in self.active_pairs:
                asyncio.create_task(pA.run_reconciliation())
                if isinstance(pB, SyncBridgePlayer):
                    asyncio.create_task(pB.run_reconciliation())
                
            if loop_idx % 6 == 0:
                for pA, pB in self.active_pairs:
                    asyncio.create_task(
                        safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")
                    )

    def release_slot(self):
        def _safe_release():
            try: self.ep_sem.release()
            except ValueError: pass
        self.async_loop.call_soon_threadsafe(_safe_release)

    def step(self, actions_dict: dict) -> List[Tuple]:
        for tag, action_idx in actions_dict.items():
            if tag in self.action_futures:
                fut, loop = self.action_futures.pop(tag)
                if not fut.done():
                    loop.call_soon_threadsafe(fut.set_result, action_idx)

        events = [self.event_queue.get()]
        time.sleep(0.003)
        
        while True:
            try: events.append(self.event_queue.get_nowait())
            except queue.Empty: break

        return events

# ---------------------------------------------------------------------------
# ROLLOUT WORKER (RAY ACTOR) - PURE SYNC MAIN THREAD
# ---------------------------------------------------------------------------

class RolloutWorker:
    def __init__(self, cfg: RunConfig, inference_actor, learner_actor, pairs: int, server_port: int):
        self.cfg = cfg
        self.started = time.perf_counter()
        self.started_cpu = time.process_time()
        self.learner_client = SyncLearnerClient(learner_actor, cfg)
        
        # --- NEW: Store the Central Inference Actor ---
        self.inference_actor = inference_actor
        
        self.vec_env = PokeEnvVectorWrapper(cfg, pairs, server_port)
        self._traj = {}
        self.teacher_labels = 0
        self.student_actions = 0
        self.teacher_actions = 0
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
            teacher_targets = []
            
            for event_type, tag, payload, extra_data in events:
                if event_type == "ERROR":
                    raise RuntimeError(f"Battle processing failed for {tag}: {payload}")
                elif event_type == "STEP":
                    step_tags.append(tag)
                    step_obs.append(payload) 
                    teacher_targets.append(None)
                elif event_type == "DAGGER":
                    step_tags.append(tag)
                    step_obs.append(payload)
                    teacher_targets.append(int(extra_data))
                elif event_type == "IMITATION":
                    traj = self._traj.setdefault(tag, {"obs": [], "act": [], "logp": [], "val": [], "var": []})
                    traj["obs"].append(payload)    # payload is obs_flat
                    traj["act"].append(extra_data)      # extra is action_idx
                    traj["logp"].append(0.0)
                    traj["val"].append(0.0)
                    traj["var"].append(0.0)
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
                    target = teacher_targets[i]
                    behavior_action = int(acts[i])
                    if target is not None:
                        self.teacher_labels += 1
                        if random.random() < self.cfg.learner.teacher_probability:
                            behavior_action = target
                            self.teacher_actions += 1
                        else:
                            self.student_actions += 1
                    actions_dict[tag] = behavior_action
                    
                    traj = self._traj.setdefault(tag, {"obs": [], "act": [], "logp": [], "val": [], "var": []})
                    traj["obs"].append(step_obs[i])
                    traj["act"].append(target if target is not None else acts[i])
                    traj["logp"].append(0.0 if target is not None else logps[i])
                    traj["val"].append(0.0 if target is not None else vals[i])
                    traj["var"].append(0.0 if target is not None else var[i])

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
                if self.cfg.learner.heuristic_opponent:
                    self.model_games += 1
                    self.model_wins += int(bool(battle.won))
                T = len(buf["act"])
                obs_stacked = np.stack(buf["obs"], axis=0)
                rewards = episode_rewards(obs_stacked, bool(battle.won), bool(battle.lost), self.cfg.reward,
                    self.cfg.learner.gamma, self.assembler.offsets, self.assembler.meta)
                dones = np.zeros(T, dtype=np.float32)
                dones[-1] = 1.0
                
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
            "teacher_labels": self.teacher_labels,
            "student_actions": self.student_actions,
            "teacher_actions": self.teacher_actions,
            "model_games": self.model_games,
            "model_wins": self.model_wins,
        }
