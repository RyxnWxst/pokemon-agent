"""README-era PPO defaults, multiplier profiles, server assignment, resume guard, stock heuristic."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from config import PPO_LR_MULTIPLIERS, ROOT, LearnerConfig, ModelConfig, RunConfig
from learner import LearnerActor
from model import make_model
from optimizers import make_optimizer
from checkpoint_io import load_weights, model_config, read_checkpoint
from ppo_core import AsyncEpisodeDataset, masked_logprob_entropy, minibatch_bounds, ppo_update
from vendor.historical_config import LearnerConfig as HistoricalLearnerConfig


class ReadmeRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.architecture = ModelConfig(**json.loads((ROOT/"tests/tiny_model.json").read_text()))

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
        from train import parse_args
        args, defaults = parse_args([]), LearnerConfig()
        self.assertEqual((args.steps_per_update, args.batch_size, args.microbatch_size, args.epochs,
                          args.learning_rate, args.lr_schedule, args.kl_stop_rule, args.lr_multipliers),
            (defaults.steps_per_update, defaults.minibatch_size, defaults.microbatch_size, defaults.update_epochs,
             defaults.lr, defaults.lr_schedule, defaults.kl_stop_rule, defaults.ppo_lr_multipliers))
        self.assertEqual((args.ports, args.worker_ports, args.worker_queue_episodes), ([8002], [8002], 0))

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

    def test_workers_are_spread_across_servers(self):
        from train import assign_ports, parse_args
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

    def learner(self, directory, **settings):
        learner = LearnerActor.__new__(LearnerActor)
        learner.cfg = replace(LearnerConfig(), device="cpu", mode="ppo", ckpt_dir=str(directory), **settings)
        learner.run_cfg = replace(RunConfig.default(), model=self.architecture, learner=learner.cfg)
        learner.net = make_model(self.architecture)
        learner.opt, learner.sched = make_optimizer(learner.net, learner.cfg)
        learner.dataset = AsyncEpisodeDataset(14, "cpu")
        learner.update_idx = learner.total_steps = learner.total_episodes = 0
        learner.directory = Path(directory)
        return learner

    def test_resume_refuses_settings_it_would_ignore(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = self.learner(temporary, ppo_lr_multipliers="current")._save_checkpoint(
                Path(temporary)/"learner_update_000000001.pt")
            for changed in (dict(ppo_lr_multipliers="historical"), dict(ppo_lr_multipliers="current", lr=3e-5),
                            dict(ppo_lr_multipliers="current", lr_schedule="constant")):
                with self.assertRaisesRegex(ValueError, "optimizer settings"):
                    self.learner(temporary, **changed)._load_checkpoint(saved)
            self.learner(temporary, ppo_lr_multipliers="current")._load_checkpoint(saved)
            # Older checkpoints predate the setting and used the current multipliers.
            checkpoint = torch.load(saved, weights_only=False)
            del checkpoint["run_cfg"]["learner"]["ppo_lr_multipliers"]
            legacy = Path(temporary)/"legacy.pt"
            torch.save(checkpoint, legacy)
            with self.assertRaisesRegex(ValueError, "ppo_lr_multipliers"):
                self.learner(temporary)._load_checkpoint(legacy)
            self.learner(temporary, ppo_lr_multipliers="current")._load_checkpoint(legacy)


    def test_small_remainders_join_the_last_minibatch(self):
        self.assertEqual(minibatch_bounds(9, 4), [(0, 4), (4, 9)])
        self.assertEqual(minibatch_bounds(10, 4), [(0, 4), (4, 8), (8, 10)])  # half a batch keeps its own step
        self.assertEqual(minibatch_bounds(8, 4), [(0, 4), (4, 8)])
        self.assertEqual(minibatch_bounds(3, 4), [(0, 3)])
        self.assertEqual(minibatch_bounds(0, 4), [])
        update = minibatch_bounds(32776, 4096)  # the pilot's first update
        self.assertEqual((len(update), update[-1]), (8, (28672, 32776)))
        for n, size in ((32776, 4096), (34816, 4096), (8, 3), (1, 4096)):
            bounds = minibatch_bounds(n, size)
            self.assertEqual([start for start, _ in bounds], [0] + [end for _, end in bounds[:-1]])
            self.assertEqual(bounds[-1][1], n)
            self.assertTrue(all(size//2 <= end-start < size + size//2 for start, end in bounds) or len(bounds) == 1)

    def test_ppo_update_merges_a_small_remainder(self):
        torch.manual_seed(71)
        model = make_model(self.architecture)
        fixture = torch.load(ROOT/"tests/fixtures/battle-observations.pt", weights_only=True)
        obs, actions = fixture["observations"], fixture["actions"]
        with torch.no_grad():
            start, end = model.unpacker.offsets["action_mask"]
            logp, _ = masked_logprob_entropy(model(obs)[0], obs[:, start:end], actions)
        n = len(actions)
        dataset = AsyncEpisodeDataset(14, "cpu")
        dataset.add_steps(obs, actions, logp, torch.zeros(n), torch.linspace(-1, 1, n), torch.linspace(1, -1, n),
            torch.zeros(n), torch.arange(n))
        cfg = replace(LearnerConfig(), device="cpu", mode="ppo", update_epochs=1, target_kl=0.,
            minibatch_size=6, microbatch_size=4)
        optimizer, _ = make_optimizer(model, cfg)
        stats = ppo_update(model, optimizer, dataset, mode="ppo", **cfg.ppo_kwargs())
        self.assertEqual((stats.n_mb, stats.samples_processed, stats.first_pass_samples), (1, n, n))

    def test_milestones_are_saved_once_and_load_for_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            learner = self.learner(temporary, milestone_every_states=100)
            learner.total_steps, learner.update_idx = 150, 3
            first = learner._maybe_save_milestone()
            self.assertTrue(first.endswith("states-000000000100.pt"))
            self.assertIsNone(learner._maybe_save_milestone())
            learner.total_steps = 250
            second = learner._maybe_save_milestone()
            self.assertTrue(second.endswith("states-000000000200.pt"))
            # A resumed learner starts past milestones it has already crossed.
            learner.total_steps, learner.last_milestone = 299, 2
            self.assertIsNone(learner._maybe_save_milestone())
            checkpoint = read_checkpoint(second)
            self.assertEqual((checkpoint["milestone_states"], checkpoint["total_steps"]), (200, 250))
            self.assertNotIn("optimizer", checkpoint)
            model = make_model(model_config(checkpoint))
            load_weights(model, checkpoint)
            for key, value in learner.net.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key])
            self.assertIsNone(self.learner(temporary, milestone_every_states=0)._maybe_save_milestone())


class StockHeuristicTests(unittest.TestCase):
    def test_stock_estimate_matches_unpatched_poke_env_source(self):
        import poke_env.player.baselines as installed
        from evaluate import stock_stat_estimation
        spec = importlib.util.spec_from_file_location("unpatched_baselines", installed.__file__)
        unpatched = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unpatched)
        original = unpatched.SimpleHeuristicsPlayer._stat_estimation
        for boost in range(-6, 7):
            mon = SimpleNamespace(boosts=dict(atk=boost), base_stats=dict(atk=97))
            self.assertEqual(stock_stat_estimation(mon, "atk"), original(mon, "atk"))

    def test_stock_player_swaps_estimate_only_during_its_decision(self):
        from evaluate import StockHeuristicsPlayer, stock_stat_estimation
        from poke_env.player import SimpleHeuristicsPlayer
        from worker import _corrected_stat_estimation
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
        self.assertIs(SimpleHeuristicsPlayer._stat_estimation, _corrected_stat_estimation)


if __name__ == "__main__":
    unittest.main()
