"""
The 14-action space: legal-action mask and the index <-> poke-env order mapping.

0-3: moves, 4-9: switch to team slot 0-5, 10-13: Terastallize and use move 0-3.
Works on poke-env objects by duck typing; nothing here imports poke-env.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np

ACTION_DIM = 14


def create_action_mask(battle: Any, self_team: List[Any]) -> np.ndarray:
    """
    Generates a valid action mask using poke-env 0.15's native engine.

    0-3: Moves, 4-9: Switches, 10-13: Terastallize + Moves.
    """
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    active_mon = battle.active_pokemon

    if not active_mon:
        return mask

    # 1. HANDLE MOVES & TERA (0-3 & 10-13)
    if battle.available_moves:
        mon_move_ids = [m.id for m in active_mon.moves.values()]
        for available_move in battle.available_moves:
            if available_move.id in mon_move_ids:
                idx = mon_move_ids.index(available_move.id)
                if idx < 4:
                    mask[idx] = 1.0
                    if battle.can_tera:
                        mask[idx + 10] = 1.0
            elif available_move.id == "struggle":
                mask[0] = 1.0

        # Safety fallback for extreme edge cases (e.g., Transform bugs)
        if not mask[:4].any():
            mask[0] = 1.0

    # 2. HANDLE SWITCHES & REVIVALS (4-9)
    # In 0.15, available_switches natively handles trapped, reviving, and fainting states.
    if battle.available_switches:
        for available_switch in battle.available_switches:
            try:
                # Map the available switch to the static 6-slot self_team array
                idx = self_team.index(available_switch)
                if idx < 6:
                    mask[4 + idx] = 1.0
            except ValueError:
                pass

    # 3. SAFETY FALLBACK
    # Ensure at least one action is always valid so the NN never deadlocks
    if not mask.any():
        mask[0] = 1.0

    return mask


def map_index_to_order(index: int, battle: Any) -> Tuple[Union[str, Any], Dict]:
    """Maps a predicted index to a valid game engine command."""
    active_mon = battle.active_pokemon
    if not active_mon:
        return "DEFAULT", {}

    # Moves and Terastallization
    if (0 <= index <= 3) or (10 <= index <= 13):
        mon_move_ids = [m.id for m in active_mon.moves.values()]
        move_idx = index if index <= 3 else index - 10

        if move_idx < len(mon_move_ids):
            target_id = mon_move_ids[move_idx]
            for am in battle.available_moves:
                if am.id == target_id:
                    return am, {"terastallize": True} if index >= 10 else {}
                elif am.id == "struggle" and move_idx == 0:
                    return am, {}

        # Fallbacks
        if battle.available_moves: return battle.available_moves[0], {}
        if battle.available_switches: return battle.available_switches[0], {}
        return "DEFAULT", {}

    # Switches
    elif 4 <= index <= 9:
        team_list = list(battle.team.values())
        slot_idx = index - 4
        if slot_idx < len(team_list) and team_list[slot_idx]:
            return team_list[slot_idx], {}

    return "DEFAULT", {}


def map_order_to_index(order: Any, battle: Any) -> int:
    """Reverse-maps a 0.15 BattleOrder to a policy index (used for Imitation Learning)."""
    if not order:
        return 0 # Handle string/DEFAULT orders
    choice = getattr(order, "order", None)
    if not choice:
        return 0
    # Case A: Move/Tera (0.15 exposes Move objects)
    if hasattr(choice, "id"):  # Safer check than base_power
        active_mon = battle.active_pokemon
        if not active_mon: return 0

        mon_move_ids = [m.id for m in active_mon.moves.values()]
        if choice.id in mon_move_ids:
            move_slot = mon_move_ids.index(choice.id)
            if getattr(order, "terastallize", False): return move_slot + 10
            return move_slot
        elif choice.id == "struggle":
            return 0
        return 0

    # Case B: Switch (0.15 exposes Pokemon objects)
    elif hasattr(choice, "current_hp"):
        team_list = list(battle.team.values())
        try:
            return 4 + team_list.index(choice)
        except ValueError:
            return 4

    return 0
