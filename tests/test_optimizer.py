"""Optimizer zones and learning-rate schedules."""
from dataclasses import replace
import unittest
import torch

from helpers import load_fixture, tiny_architecture
from pokemon_agent.config import LearnerConfig
from pokemon_agent.model.transformer import make_model
from pokemon_agent.training.optimizer import learning_rate_scale, make_optimizer


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.fixture = load_fixture()
        cls.architecture = tiny_architecture()

    def setUp(self):
        torch.manual_seed(71)
        self.model = make_model(self.architecture).eval()
        self.obs = self.fixture["observations"]
        self.actions = self.fixture["actions"]
        self.cfg = replace(LearnerConfig(), device="cpu", mode="ppo", lr=.001, lr_warmup_steps=0,
            update_epochs=1, minibatch_size=3, target_kl=0.)

    def test_constant_schedule_starts_at_full_learning_rate(self):
        cfg = replace(self.cfg, lr_schedule="constant", lr_warmup_steps=1000, ppo_lr_multipliers="current")
        optimizer, scheduler = make_optimizer(self.model, cfg)
        self.assertEqual(scheduler.get_last_lr(), scheduler.base_lrs)
        self.assertEqual(max(scheduler.get_last_lr()), 2*cfg.lr)
        for scale in scheduler.lr_lambdas:
            self.assertEqual([scale(step) for step in (0, 100, 1000, 500000)], [1., 1., 1., 1.])

    def test_constant_learning_rate_does_not_depend_on_optimizer_step_count(self):
        cfg = replace(self.cfg, lr_schedule="constant", lr_warmup_steps=1000)
        optimizer, scheduler = make_optimizer(self.model, cfg)
        initial = [group["lr"] for group in optimizer.param_groups]
        self.assertTrue(all(rate > 0 for rate in initial))
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        self.assertEqual(initial, [group["lr"] for group in optimizer.param_groups])

    def test_multiplier_profiles_set_zone_learning_rates(self):
        model = make_model(self.architecture)
        expected = {"historical": dict(pi=1., v=1., other=1.), "current": dict(pi=1., v=2., other=.5)}
        for profile, rates in expected.items():
            cfg = replace(LearnerConfig(), device="cpu", mode="ppo", ppo_lr_multipliers=profile)
            optimizer, _ = make_optimizer(model, cfg)
            for group in optimizer.param_groups:
                scale = rates.get(group["name"], rates["other"])
                # The scheduler has already applied warmup to "lr"; the configured rate is initial_lr.
                self.assertAlmostEqual(group["initial_lr"], cfg.lr*scale, msg=f"{profile} {group['name']}")
        with self.assertRaisesRegex(ValueError, "multipliers"):
            LearnerConfig(ppo_lr_multipliers="unknown")

    def test_learning_rate_scale_warms_holds_and_decays(self):
        cfg = replace(LearnerConfig(), lr_warmup_steps=10, lr_hold_steps=20, lr_total_steps=130)
        self.assertEqual([learning_rate_scale(step, cfg) for step in (0, 4, 9)], [.1, .5, 1.])
        self.assertEqual([learning_rate_scale(step, cfg) for step in (10, 29, 30)], [1., 1., 1.])
        self.assertAlmostEqual(learning_rate_scale(80, cfg), 1/5**1.5)
        self.assertAlmostEqual(learning_rate_scale(130, cfg), 1/27)
        self.assertAlmostEqual(learning_rate_scale(10**6, cfg), 1/27)
        rates = [learning_rate_scale(step, cfg) for step in range(30, 131)]
        self.assertTrue(all(later <= earlier for earlier, later in zip(rates, rates[1:])))
        constant = replace(cfg, lr_schedule="constant")
        self.assertEqual({learning_rate_scale(step, constant) for step in (0, 15, 80, 10**6)}, {1.})


if __name__ == "__main__":
    unittest.main()
