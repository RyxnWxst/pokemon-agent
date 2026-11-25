"""
Observation layout: the nine blocks of the flat vector, their sizes and offsets.

Pure Python: nothing here imports poke-env, so the model can learn the layout
without loading the battle client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pokemon_agent.obs.actions import ACTION_DIM
from pokemon_agent.obs.moves import get_move_metadata
from pokemon_agent.obs.pokemon import get_pokemon_metadata
from pokemon_agent.obs.transitions import transition_id_dim, transition_scalar_dim
from pokemon_agent.obs.vocab import load_vocab

# Block names in the order they appear in the flat observation vector.
BLOCKS = (
    "pokemon_body", "pokemon_ids", "ability_ids", "move_ids", "move_scalars",
    "global_scalars", "transition_ids", "transition_scalars", "action_mask",
)


def get_schema_metadata(vocab_lists: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    if vocab_lists is None:
        vocab_lists = load_vocab()
    move_meta = get_move_metadata(vocab_lists)
    pok_meta = get_pokemon_metadata(vocab_lists)

    v_type = len(vocab_lists.get("pokemon.type", [])) + 1

    return {
        "dim_pokemon_body": pok_meta["scalar_dim"],
        "dim_move_scalars": move_meta["single_move_dim"] * 4,
        "dim_transition_ids": transition_id_dim(),
        "dim_transition_scalars": transition_scalar_dim(),
        "dim_global_scalars": 3 + (len(vocab_lists.get("global.weather", [])) + 1 + 10) +
                              (len(vocab_lists.get("global.field", [])) + 1) +
                              ((len(vocab_lists.get("global.side_condition", [])) + 1) * 2),
        "feature_map": {
            "body": pok_meta["feature_map"],
            "move": move_meta["feature_map"],
            "global": {"turn_int": 0, "remainder_raw": (1, None)},
            "transition": {
                "move_ids": (0, 2), "pokemon_ids": (2, 6), "ability_ids": (6, 8), "item_ids": (8, 10),
            },
        },
        "faint_internal_idx": pok_meta["faint_internal_idx"], # Index 113
        "vocab_pokemon": len(vocab_lists.get("pokemon.species", [])) + 1,
        "vocab_item": len(vocab_lists.get("pokemon.item", [])) + 1,
        "vocab_ability": len(vocab_lists.get("pokemon.ability", [])) + 1,
        "vocab_move": len(vocab_lists.get("move.id", [])) + 1,
        "vocab_type": v_type,
        "action_dim": ACTION_DIM,
        "n_pokemon_slots": 12, "n_move_slots": 4, "n_ability_slots": 4,
        "n_transition_ids": transition_id_dim(),
    }


def block_offsets(meta: Dict[str, Any]) -> Tuple[List[Tuple[str, Tuple[int, ...], int]], Dict[str, Tuple[int, int]], int]:
    """Returns (layout, offsets, total_dim) for the nine blocks: name, shape, size; start/end; total size."""
    layout = [
        ("pokemon_body", (12, meta["dim_pokemon_body"]), 12 * meta["dim_pokemon_body"]),
        ("pokemon_ids", (12, 2), 24),
        ("ability_ids", (12, 4), 48),
        ("move_ids", (12, 4), 48),
        ("move_scalars", (12, meta["dim_move_scalars"]), 12 * meta["dim_move_scalars"]),
        ("global_scalars", (meta["dim_global_scalars"],), meta["dim_global_scalars"]),
        ("transition_ids", (meta["dim_transition_ids"],), meta["dim_transition_ids"]),
        ("transition_scalars", (meta["dim_transition_scalars"],), meta["dim_transition_scalars"]),
        ("action_mask", (meta["action_dim"],), meta["action_dim"]),
    ]
    assert tuple(name for name, _, _ in layout) == BLOCKS
    offsets = {}
    curr = 0
    for name, _, size in layout:
        offsets[name] = (curr, curr + size)
        curr += size
    return layout, offsets, curr


def observation_metadata(vocab_lists: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Schema metadata plus block offsets: what the model needs to read an observation."""
    meta = get_schema_metadata(vocab_lists)
    _, offsets, _ = block_offsets(meta)
    meta["offsets"] = offsets
    return meta


def translate_relative_index(meta: Dict[str, Any], block_name: str, relative_idx: int) -> str:
    """Translates a raw block index into a human-readable feature name (a debugging aid)."""
    try:
        if block_name == "pokemon_body":
            single_dim = meta["dim_pokemon_body"]
            slot = relative_idx // single_dim
            idx_in_mon = relative_idx % single_dim

            for k, v in meta["feature_map"]["body"].items():
                if isinstance(v, tuple) and v[0] <= idx_in_mon < v[1]:
                    return f"P{slot//6 + 1}_Mon{slot%6}.{k}[{idx_in_mon - v[0]}]"
                elif v == idx_in_mon:
                    return f"P{slot//6 + 1}_Mon{slot%6}.{k}"

        elif block_name == "move_scalars":
            # dim_move_scalars is the total size for all 4 moves of ONE pokemon
            single_move_dim = meta["dim_move_scalars"] // 4
            slot = relative_idx // meta["dim_move_scalars"]
            idx_in_mon = relative_idx % meta["dim_move_scalars"]
            move_num = idx_in_mon // single_move_dim
            idx_in_move = idx_in_mon % single_move_dim

            for k, v in meta["feature_map"]["move"].items():
                if isinstance(v, tuple) and v[0] <= idx_in_move < v[1]:
                    return f"P{slot//6 + 1}_Mon{slot%6}.Move{move_num}.{k}[{idx_in_move - v[0]}]"
                elif v == idx_in_move:
                    return f"P{slot//6 + 1}_Mon{slot%6}.Move{move_num}.{k}"

        elif block_name == "global_scalars":
            if relative_idx == 0: return "turn_int"
            if relative_idx == 1: return "p1_tera_flag"
            if relative_idx == 2: return "p2_tera_flag"
            return f"weather_or_side_condition[{relative_idx}]"

        elif block_name == "transition_ids":
            names = ["P1_Move", "P2_Move", "P1_Actor", "P1_Target", "P2_Actor", "P2_Target", "P1_Abil", "P2_Abil", "P1_Item", "P2_Item"]
            return names[relative_idx] if relative_idx < len(names) else f"TransID[{relative_idx}]"

        elif block_name == "transition_scalars":
            side = "P1" if relative_idx < 45 else "P2"
            idx = relative_idx % 45
            return f"{side}_Transition_Scalar[{idx}]"

    except Exception:
        pass

    return f"Unknown[{relative_idx}]"
