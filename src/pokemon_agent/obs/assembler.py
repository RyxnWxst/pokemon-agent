"""
Observation Assembler for Pokémon Showdown Reinforcement Learning.

This module converts complex Battle objects into structured, flattened NumPy
arrays suitable for neural network input. The layout lives in layout.py and
the action mask and order mapping in actions.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from pokemon_agent.obs.abilities import encode_ability_inplace
from pokemon_agent.obs.actions import create_action_mask
from pokemon_agent.obs.global_state import encode_global_inplace
from pokemon_agent.obs.layout import block_offsets, get_schema_metadata
from pokemon_agent.obs.moves import encode_moves_inplace
from pokemon_agent.obs.poke_env_compat import apply_request_fixes
from pokemon_agent.obs.pokemon import encode_pokemon_body_inplace
from pokemon_agent.obs.transitions import encode_transitions_inplace
from pokemon_agent.obs.vocab import build_vocab_map, load_vocab

# Configure logging
logger = logging.getLogger(__name__)


class ObservationAssembler:
    """
    Orchestrates the conversion of Battle states into numerical observations.

    Building one also applies the poke-env request fix that keeps your own
    Pokémon's Tera type (idempotent; it used to run on import).
    """

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None):
        """
        Initializes the assembler with vocabulary mappings and schema metadata.

        Args:
            vocab: A dictionary containing lists of species, items, moves, etc.
                  If None, loads the bundled vocab.json.
        """
        apply_request_fixes()
        if vocab is None:
            vocab = load_vocab()

        self.vocab_lists = vocab
        self.vocab_map = build_vocab_map(vocab)

        # Metadata defines the dynamic sizes of sub-vectors (e.g., number of types)
        self.meta = get_schema_metadata(vocab)

        self.pokemon_scalar_dim = self.meta["dim_pokemon_body"]
        self.move_scalar_dim = self.meta["dim_move_scalars"]
        self.global_dim = self.meta["dim_global_scalars"]
        self.transition_id_dim = self.meta["dim_transition_ids"]
        self.transition_scalar_dim = self.meta["dim_transition_scalars"]
        self.action_dim = self.meta["action_dim"]

        # Layout of the flattened observation vector
        self.layout, self.offsets, self.total_dim = block_offsets(self.meta)
        self.meta["offsets"] = self.offsets
        self.calc_buf = np.zeros(self.total_dim, dtype=np.float16)

    def assemble(self, battle: Any, events: List[List[str]] = None) -> np.ndarray:
        """
        Converts a Battle state into a flattened NumPy array.

        Args:
            battle: A poke_env Battle object.

        Returns:
            np.ndarray: A flattened vector of shape (total_dim,).
        """
        self.calc_buf.fill(0)
        off = self.offsets

        # Extract teams (ensuring internal server consistency)
        self_team = list(battle.team.values())
        opp_team = list(battle.opponent_team.values())

        # 1. ENCODE POKEMON (12 Slots: 6 Self, 6 Opponent)
        for i in range(12):
            is_self = i < 6
            mon = self_team[i] if (is_self and i < len(self_team)) else \
                  (opp_team[i-6] if (not is_self and (i-6) < len(opp_team)) else None)

            # Encode physical properties, stats, and health
            encode_pokemon_body_inplace(
                mon, self.calc_buf, i, self.pokemon_scalar_dim,
                off, self.vocab_map, self.vocab_lists
            )

            # Encode categorical IDs (Species, Item, Ability)
            encode_ability_inplace(mon, self.calc_buf, i, off, self.vocab_map)

            # Encode categorical IDs (Species, Item, Ability)
            available_move_ids = []
            if is_self and mon and mon.active and battle.available_moves:
                available_move_ids = [m.id for m in battle.available_moves]

            # Encode individual move properties
            encode_moves_inplace(
                mon, self.calc_buf, i, self.move_scalar_dim,
                off, self.vocab_map, self.vocab_lists,
                is_self=is_self, # Pass the boolean directly
                available_move_ids=available_move_ids
            )

        # 2. ENCODE GLOBAL STATE (Weather, Terrain, Hazards)
        encode_global_inplace(
            battle, self.calc_buf, off["global_scalars"],
            self.vocab_map, self.vocab_lists
        )

        # 3. ENCODE TRANSITIONS (Move History & Type Effectiveness)

        player_role = getattr(battle, "player_role", None)
        if player_role not in ("p1", "p2"):
            player_role = "p1"  # safe fallback; you can also raise here if preferred

        encode_transitions_inplace(
            events,
            self.calc_buf,
            off["transition_ids"][0],
            off["transition_scalars"][0],
            self.vocab_map,
            self_is_p1=(player_role == "p1"),
        )

        # 4. ENCODE ACTION MASK
        m_start, m_end = off["action_mask"]
        self.calc_buf[m_start:m_end] = create_action_mask(battle, self_team)

        return self.calc_buf.copy()
