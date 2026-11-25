"""Messages for finished, forgotten battle rooms must not leave poke-env handlers waiting forever."""
import asyncio
import concurrent.futures
import logging
import queue
import unittest

from poke_env.battle import Battle
from poke_env.concurrency import POKE_LOOP
from poke_env.player import SimpleHeuristicsPlayer

from pokemon_agent.config import RunConfig
from pokemon_agent.env.bridge import SyncBridgePlayer
from pokemon_agent.env.heuristics import TrainingHeuristicPlayer
from pokemon_agent.env.rooms import FINISHED_ROOM_TTL_S

TAG = "battle-gen9randombattle-101"
DEINIT = [[f">{TAG}"], ["", "deinit"]]


def battle(player):
    return Battle(TAG, player.username, logging.getLogger("test"), gen=9)


def run_in_poke_loop(coroutine, timeout=1.):
    """Run on poke-env's own loop, where its conditions and queues live, as in training."""
    future = asyncio.run_coroutine_threadsafe(coroutine, POKE_LOOP)
    try:
        return future.result(timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise


class FinishedRoomTests(unittest.TestCase):
    def bridge(self):
        return SyncBridgePlayer(cfg=RunConfig.default(), event_queue=queue.Queue(), action_futures={},
            ep_sem=None, battle_format="gen9randombattle", start_listening=False)

    def test_unpatched_poke_env_waits_forever_for_a_forgotten_battle(self):
        """Documents the original failure: one stuck handler per finished battle."""
        player = SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
        with self.assertRaises(concurrent.futures.TimeoutError):
            run_in_poke_loop(player._handle_battle_message(DEINIT), timeout=.3)

    def test_rollout_player_ignores_deinit_after_finishing(self):
        player = self.bridge()
        finished = battle(player)
        player._battles[TAG] = finished
        player._battle_events[TAG] = [["", "move"]]
        player._battle_finished_callback(finished)
        self.assertNotIn(TAG, player._battles)
        self.assertIn(TAG, player._finished_rooms)
        self.assertEqual(player.event_queue.get_nowait()[0], "DONE")

        # A trailing message is dropped too, then |deinit closes the record.
        run_in_poke_loop(player._handle_battle_message([[f">{TAG}"], ["", "raw", "x"]]))
        self.assertIn(TAG, player._finished_rooms)
        run_in_poke_loop(player._handle_battle_message(DEINIT))
        self.assertNotIn(TAG, player._finished_rooms)
        self.assertNotIn(TAG, player._battle_events)

    def test_cleanup_of_stalled_battles_also_filters_their_room(self):
        player = self.bridge()
        tag = f"{player.username}_{TAG}"
        player._battles[TAG] = battle(player)
        player._cleanup_local_battle(tag)
        self.assertIn(TAG, player._finished_rooms)
        run_in_poke_loop(player._handle_battle_message(DEINIT))
        self.assertNotIn(TAG, player._finished_rooms)

    def test_teacher_opponent_ignores_deinit_after_finishing(self):
        player = TrainingHeuristicPlayer(battle_format="gen9randombattle", start_listening=False)
        finished = battle(player)
        player._battles[TAG] = finished
        player._battle_finished_callback(finished)
        run_in_poke_loop(player._handle_battle_message(DEINIT))
        self.assertEqual((len(player._battles), len(player._finished_rooms)), (0, 0))

    def test_records_expire_when_deinit_never_arrives(self):
        player = self.bridge()
        player._finished_rooms["battle-gen9randombattle-1"] = -FINISHED_ROOM_TTL_S
        player._forget_room(TAG)
        self.assertEqual(list(player._finished_rooms), [TAG])

    def test_active_rooms_are_not_filtered(self):
        player = self.bridge()
        self.assertFalse(player._is_finished_room([[f">{TAG}"], ["", "turn", "3"]]))
        self.assertFalse(player._is_finished_room([["", "updateuser", "x"]]))
        self.assertEqual(player.pending_message_tasks(), 0)


if __name__ == "__main__":
    unittest.main()
