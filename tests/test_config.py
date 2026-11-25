"""README-era defaults, the training command line, and package self-containment."""
import importlib
import pkgutil
from pathlib import Path
import unittest
from unittest.mock import patch

import pokemon_agent
from pokemon_agent.config import PPO_LR_MULTIPLIERS, LearnerConfig
from pokemon_agent.obs.assembler import ObservationAssembler
from pokemon_agent.paths import PACKAGE_DIR
from pokemon_agent.vendor.historical_config import LearnerConfig as HistoricalLearnerConfig


class ConfigTests(unittest.TestCase):
    def test_learner_defaults_match_de8cc69(self):
        current, historical = LearnerConfig(), HistoricalLearnerConfig(mode="ppo")
        for name in ("steps_per_update", "minibatch_size", "update_epochs", "lr", "lr_warmup_steps",
                     "lr_hold_steps", "lr_total_steps", "weight_decay", "gamma", "gae_lambda", "clip_coef",
                     "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "v_min", "v_max", "v_bins"):
            self.assertEqual(getattr(current, name), getattr(historical, name), name)
        self.assertEqual((current.kl_stop_rule, current.lr_schedule, current.ppo_lr_multipliers),
            ("historical", "historical", "historical"))
        self.assertEqual(tuple(PPO_LR_MULTIPLIERS["historical"].values()),
            (historical.lr_backbone_mult, historical.lr_pi_mult, historical.lr_v_mult))
        self.assertFalse(current.use_dynamic_lambda)

    def test_training_cli_defaults_are_the_learner_defaults(self):
        from pokemon_agent.cli.train import parse_args
        args, defaults = parse_args([]), LearnerConfig()
        self.assertEqual((args.steps_per_update, args.batch_size, args.microbatch_size, args.epochs,
                          args.learning_rate, args.lr_schedule, args.kl_stop_rule, args.lr_multipliers),
            (defaults.steps_per_update, defaults.minibatch_size, defaults.microbatch_size, defaults.update_epochs,
             defaults.lr, defaults.lr_schedule, defaults.kl_stop_rule, defaults.ppo_lr_multipliers))
        self.assertEqual((args.ports, args.worker_ports, args.worker_queue_episodes), ([8002], [8002], 0))

    def test_workers_are_spread_across_servers(self):
        from pokemon_agent.cli.train import assign_ports, parse_args
        self.assertEqual(assign_ports(5, [8010, 8011]), [8010, 8011, 8010, 8011, 8010])
        with self.assertRaisesRegex(ValueError, "one rollout worker per"):
            assign_ports(1, [8010, 8011])
        args = parse_args(["--ports", "8010", "8011", "--workers", "4", "--concurrency", "128"])
        self.assertEqual((args.ports, args.worker_ports), ([8010, 8011], [8010, 8011, 8010, 8011]))
        with patch("sys.stderr"):
            for invalid in (["--ports", "8010", "8010", "--workers", "2", "--concurrency", "8"],
                            ["--ports", "8010", "8011"], ["--worker-queue-episodes", "-1"]):
                with self.assertRaises(SystemExit):
                    parse_args(invalid)

    def test_imports_are_self_contained(self):
        names = [module.name for module in pkgutil.walk_packages(pokemon_agent.__path__, "pokemon_agent.")]
        self.assertGreater(len(names), 40)
        for name in names:
            self.assertTrue(Path(importlib.import_module(name).__file__).resolve().is_relative_to(PACKAGE_DIR), name)
        self.assertTrue(ObservationAssembler().total_dim > 0)


if __name__ == "__main__":
    unittest.main()
