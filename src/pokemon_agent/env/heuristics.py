"""The heuristic teacher and opponent: corrected and stock stat estimates and their players."""
import math

from poke_env.player import SimpleHeuristicsPlayer

from pokemon_agent.env.rooms import FinishedRoomFilter


def corrected_stat_estimation(mon, stat: str) -> float:
    """
    A mathematically accurate stat estimator for Gen 9 Random Battles.
    Replaces the hallucinated Level 100 / 0 EV logic in the original bot.
    """
    # 1. FIX THE BOOST MULTIPLIER
    # The original checked `> 1`, which sent +1 down the negative branch: 2/(2-1) = 2x instead of 1.5x.
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


def stock_stat_estimation(mon, stat):
    """poke-env 0.15 SimpleHeuristicsPlayer._stat_estimation, verbatim (before the April 2026 correction)."""
    # Stats boosts value
    if mon.boosts[stat] > 1:
        boost = (2 + mon.boosts[stat]) / 2
    else:
        boost = 2 / (2 - mon.boosts[stat])
    return ((2 * mon.base_stats[stat] + 31) + 5) * boost


class StockHeuristicsPlayer(SimpleHeuristicsPlayer):
    """The heuristic with poke-env's original stat estimate, as in the README-era evaluation.

    The corrected estimate is patched onto the shared class and the static move
    chooser looks it up there, so it is swapped only for this player's synchronous
    decision. Tera parsing (poke_env_compat) remains process-wide for both sides.
    """
    def choose_move(self, battle):
        corrected = SimpleHeuristicsPlayer.__dict__["_stat_estimation"]
        SimpleHeuristicsPlayer._stat_estimation = staticmethod(stock_stat_estimation)
        try:
            return super().choose_move(battle)
        finally:
            SimpleHeuristicsPlayer._stat_estimation = corrected


class TrainingHeuristicPlayer(FinishedRoomFilter, SimpleHeuristicsPlayer):
    """The corrected heuristic as a training opponent (ppo_vs_teacher), ignoring finished rooms."""
    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self._forget_room(battle.battle_tag)


OPPONENTS = {"corrected": SimpleHeuristicsPlayer, "stock": StockHeuristicsPlayer}
