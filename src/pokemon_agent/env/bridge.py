"""
The thread bridge between poke-env and the synchronous rollout loop.

SyncBridgePlayer turns each decision into a STEP event on a thread-safe queue and
awaits a future the rollout thread resolves with an action index.
PokeEnvVectorWrapper runs the players' asyncio loop in a background thread,
starts battles through the rlspawn plugin, and keeps them healthy.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import secrets
import threading
import time
from typing import Any, Final, List, Tuple

from poke_env import AccountConfiguration
from poke_env.player import DefaultBattleOrder
from poke_env.player.baselines import SimpleHeuristicsPlayer

from pokemon_agent.config import RunConfig
from pokemon_agent.env.events import ProtocolEventRecorder
from pokemon_agent.env.heuristics import TrainingHeuristicPlayer
from pokemon_agent.env.rooms import FinishedRoomFilter
from pokemon_agent.env.showdown import (battle_tag_for, join_lobby, make_server_conf, mk_name,
    rlautospawn_command, safe_send, wait_for_login_retry)
from pokemon_agent.obs.actions import map_index_to_order, map_order_to_index
from pokemon_agent.obs.assembler import ObservationAssembler

ZOMBIE_LIMIT_S: Final[float] = 300.0
STALL_LIMIT_S: Final[float] = 60.0
REPAIR_INTERVAL_S: Final[float] = 30.0

logger = logging.getLogger(__name__)


class SyncBridgePlayer(FinishedRoomFilter, ProtocolEventRecorder, SimpleHeuristicsPlayer):
    def __init__(self, cfg: RunConfig, event_queue: queue.Queue, action_futures: dict, ep_sem: asyncio.Semaphore, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.event_queue = event_queue
        self.action_futures = action_futures
        self.ep_sem = ep_sem
        self.assembler = ObservationAssembler()
        self._battle_starts = {}
        self._last_act_time = {}
        self._episode_slot_state = {}

    def _get_unique_tag(self, battle):
        return f"{self.username}_{battle_tag_for(battle)}"

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

            obs_flat = self.assembler.assemble(battle, events=self.pop_events(battle.battle_tag))
            if self.cfg.learner.mode == "imitation":
                return self._teacher_step(battle, tag, obs_flat)

            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self.action_futures[tag] = (fut, loop)
            self.event_queue.put_nowait(("STEP", tag, obs_flat, battle))

            action_idx = await fut

            action_obj, kwargs = map_index_to_order(action_idx, battle)
            if action_obj == "DEFAULT": return DefaultBattleOrder()
            return self.create_order(action_obj, **kwargs)

        except Exception as e:
            logger.error(f"SyncBridgePlayer crashed for {battle.battle_tag}: {e}")
            self.event_queue.put_nowait(("ERROR", tag if 'tag' in locals() else battle.battle_tag, str(e), None))
            return self.choose_random_move(battle)

    def _teacher_step(self, battle, tag, obs_flat):
        """Online imitation: play the heuristic's move and record it as the label."""
        order = super().choose_move(battle)
        action_idx = map_order_to_index(order, battle)
        start, end = self.assembler.offsets["action_mask"]
        if obs_flat[start:end][action_idx] < 0.5:
            raise RuntimeError("Teacher action is outside the legal action mask")

        # Send directly to the trajectory compiler
        self.event_queue.put_nowait(("IMITATION", tag, obs_flat, action_idx))

        action_obj, kwargs = map_index_to_order(action_idx, battle)
        if action_obj == "DEFAULT": return DefaultBattleOrder()
        return self.create_order(action_obj, **kwargs)

    def _battle_finished_callback(self, battle):
        tag = self._get_unique_tag(battle)
        self.event_queue.put_nowait(("DONE", tag, None, battle))

        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)

        self.action_futures.pop(tag, None)
        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        self._forget_room(base_tag)
        self.drop_events(base_tag)

        super()._battle_finished_callback(battle)

    def _cleanup_local_battle(self, tag: str):
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)

        self.action_futures.pop(tag, None)

        self.event_queue.put_nowait(("CLEANUP", tag, None, None))

        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        self._forget_room(base_tag)
        self.drop_events(base_tag)

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
            self._reconcile(server_ids)

    def _reconcile(self, server_ids: set) -> None:
        """Rejoin battles only the server knows, rescue stalled ones, drop ones the server has forgotten."""
        local_map = {tag: (tag.split("_", 1)[1] if "_" in tag else tag) for tag in self._battle_starts.keys()}
        local_base_ids = set(local_map.values())

        ghosts = list(server_ids - local_base_ids)
        for bid in ghosts:
            asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))

        now = time.time()
        stalls = [tag for tag, bid in local_map.items()
            if bid in server_ids and (now - self._last_act_time.get(tag, 0)) > STALL_LIMIT_S]
        for tag in stalls:
            bid = local_map[tag]
            self._cleanup_local_battle(tag)
            asyncio.create_task(self.ps_client.send_message(f"/rlrescue {bid}", room=""))
            asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))

        zombies = [tag for tag, bid in local_map.items()
            if bid not in server_ids and (now - self._battle_starts.get(tag, 0)) > ZOMBIE_LIMIT_S]
        for tag in zombies:
            self._cleanup_local_battle(tag)


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
            if self.cfg.learner.mode == "ppo_vs_teacher":
                pB = TrainingHeuristicPlayer(**opponent_args)
            else:
                pB = SyncBridgePlayer(cfg=self.cfg, event_queue=self.event_queue,
                    action_futures=self.action_futures, ep_sem=self.ep_sem, **opponent_args)
            self.active_pairs.append((pA, pB))

        players = [p for pair in self.active_pairs for p in pair]

        for p in players:
            await wait_for_login_retry(p)
            await asyncio.sleep(0.05)

        await asyncio.gather(*[join_lobby(p) for p in players])

        for (pA, pB) in self.active_pairs:
            await safe_send(pA, rlautospawn_command(pA.username, pB.username, self.cfg.env.battle_format,
                self.cfg.rollout.rooms_per_pair), room="lobby")

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
                    asyncio.create_task(safe_send(pA, rlautospawn_command(pA.username, pB.username,
                        self.cfg.env.battle_format, self.cfg.rollout.rooms_per_pair), room="lobby"))

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
