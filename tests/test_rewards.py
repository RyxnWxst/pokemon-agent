"""Trajectory rewards: faint shaping credited to the decision before it, plus the terminal result."""
import unittest
import numpy as np

from pokemon_agent.config import RewardConfig
from pokemon_agent.obs.layout import observation_metadata
from pokemon_agent.rl.rewards import faint_counts, trajectory_rewards


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = observation_metadata()
        cls.total = max(end for _, end in cls.meta["offsets"].values())

    def observations(self, fainted_by_step):
        """One observation per step; fainted_by_step[t] lists the fainted slots (0-5 yours, 6-11 theirs)."""
        start, end = self.meta["offsets"]["pokemon_body"]
        width = self.meta["dim_pokemon_body"]
        observations = np.zeros((len(fainted_by_step), self.total), dtype=np.float16)
        for step, slots in enumerate(fainted_by_step):
            for slot in slots:
                observations[step, start + slot*width + self.meta["faint_internal_idx"]] = 1.
        return observations

    def test_faints_and_terminal_reward(self):
        # Step 1→2: an opponent faints; step 2→3: one of yours and another opponent's.
        observations = self.observations([[], [], [6], [0, 6, 7]])
        own, opponent = faint_counts(observations, self.meta)
        self.assertEqual((own.tolist(), opponent.tolist()), ([0, 0, 0, 1], [0, 0, 1, 2]))
        cfg = RewardConfig()
        rewards, dones = trajectory_rewards(observations, won=True, lost=False, reward_cfg=cfg, meta=self.meta)
        np.testing.assert_allclose(rewards, [0., cfg.faint_opp, cfg.faint_self+cfg.faint_opp, cfg.terminal_win],
            rtol=1e-6)
        self.assertEqual(dones.tolist(), [0., 0., 0., 1.])
        lost, _ = trajectory_rewards(observations, won=False, lost=True, reward_cfg=cfg, meta=self.meta)
        self.assertAlmostEqual(float(lost[-1]), cfg.terminal_loss)
        tied, _ = trajectory_rewards(observations, won=False, lost=False, reward_cfg=cfg, meta=self.meta)
        self.assertEqual(float(tied[-1]), 0.)

    def test_faint_after_last_decision_earns_nothing(self):
        # The opponent's last Pokémon faints after the final decision: only the terminal reward counts.
        observations = self.observations([[6, 7, 8, 9, 10], [6, 7, 8, 9, 10]])
        cfg = RewardConfig()
        rewards, _ = trajectory_rewards(observations, won=True, lost=False, reward_cfg=cfg, meta=self.meta)
        np.testing.assert_allclose(rewards, [0., cfg.terminal_win])
        no_shaping = RewardConfig(use_faint_reward=False)
        rewards, _ = trajectory_rewards(self.observations([[], [6], [0, 6]]), won=False, lost=True,
            reward_cfg=no_shaping, meta=self.meta)
        np.testing.assert_allclose(rewards, [0., 0., no_shaping.terminal_loss])


if __name__ == "__main__":
    unittest.main()
