"""Ignore protocol messages for battle rooms a player has already finished and forgotten."""
import time
from typing import Final

FINISHED_ROOM_TTL_S: Final[float] = 600.0


class FinishedRoomFilter:
    """Drop messages for battle rooms this player has finished and forgotten.

    Rollout players remove finished battles from poke-env's table to bound memory.
    poke-env then leaves the room, and Showdown answers with a final |deinit for it.
    For a battle missing from its table, poke-env's _get_battle waits forever, and
    every later battle start wakes all such waiters, so per-decision CPU grew with
    the number of battles a worker had finished. Those messages are now ignored.

    Place this mixin left of the poke-env player class so its
    _handle_battle_message runs first.
    """
    def __init__(self, *args, **kwargs):
        # Set before poke-env's __init__, which may start delivering messages.
        self._finished_rooms = {}
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        if self._is_finished_room(split_messages):
            return
        await super()._handle_battle_message(split_messages)

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
