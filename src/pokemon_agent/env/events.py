"""Collect each battle room's protocol lines between decisions, for the transition encoder."""
from typing import List, Optional

# Structural lines the transition encoder does not need.
_SKIPPED_COMMANDS = ("t:", "request", "turn")


def room_id_of(split_messages) -> Optional[str]:
    """The battle room a batch of split protocol lines belongs to, or None."""
    if split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
        return split_messages[0][0][1:].strip()
    return None


def transition_lines(split_messages) -> List[List[str]]:
    """Protocol lines worth keeping, exactly as obs.transitions expects them."""
    # Valid protocol lines start with an empty string (from the '|' split)
    return [parts for parts in split_messages
        if len(parts) > 1 and parts[0] == "" and parts[1] not in _SKIPPED_COMMANDS]


class ProtocolEventRecorder:
    """Mixin that stores each room's protocol lines until the next decision pops them.

    Place it left of the poke-env player class so it sees messages before poke-env.
    """
    def __init__(self, *args, **kwargs):
        self._battle_events = {}
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        room_id = room_id_of(split_messages)
        if room_id:
            lines = transition_lines(split_messages)
            if lines:
                self._battle_events.setdefault(room_id, []).extend(lines)
        # Pass to parent so poke-env can update its internal Battle object
        await super()._handle_battle_message(split_messages)

    def pop_events(self, battle_tag: str) -> List[List[str]]:
        """Events since this room's previous decision; clears them."""
        return self._battle_events.pop(battle_tag, [])

    def drop_events(self, battle_tag: str) -> None:
        self._battle_events.pop(battle_tag, None)
