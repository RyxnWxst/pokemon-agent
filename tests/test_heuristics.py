"""The stock heuristic opponent: poke-env's original stat estimate, swapped in per decision."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokemon_agent.env.patches import use_corrected_heuristic


class StockHeuristicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Commands apply this at start-up; importing the package no longer patches poke-env.
        use_corrected_heuristic()

    def test_stock_estimate_matches_unpatched_poke_env_source(self):
        import poke_env.player.baselines as installed
        from pokemon_agent.env.heuristics import stock_stat_estimation
        spec = importlib.util.spec_from_file_location("unpatched_baselines", installed.__file__)
        unpatched = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unpatched)
        original = unpatched.SimpleHeuristicsPlayer._stat_estimation
        for boost in range(-6, 7):
            mon = SimpleNamespace(boosts=dict(atk=boost), base_stats=dict(atk=97))
            self.assertEqual(stock_stat_estimation(mon, "atk"), original(mon, "atk"))

    def test_stock_player_swaps_estimate_only_during_its_decision(self):
        from pokemon_agent.env.heuristics import StockHeuristicsPlayer, stock_stat_estimation
        from poke_env.player import SimpleHeuristicsPlayer
        from pokemon_agent.env.heuristics import corrected_stat_estimation
        seen = []

        def record(battle):
            seen.append(SimpleHeuristicsPlayer._stat_estimation)
            if battle == "fail":
                raise RuntimeError("decision failed")
            return "order", 0

        player = StockHeuristicsPlayer(start_listening=False)
        with patch.object(SimpleHeuristicsPlayer, "choose_singles_move", staticmethod(record)):
            self.assertEqual(player.choose_move("battle"), "order")
            with self.assertRaisesRegex(RuntimeError, "decision failed"):
                player.choose_move("fail")
        self.assertEqual(seen, [stock_stat_estimation, stock_stat_estimation])
        self.assertIs(SimpleHeuristicsPlayer._stat_estimation, corrected_stat_estimation)


if __name__ == "__main__":
    unittest.main()
