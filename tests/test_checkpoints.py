"""Atomic status files and learner checkpoints: resume, phase changes, setting guards, milestones."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import torch

from helpers import bare_learner, fixture_dataset, load_fixture, tiny_architecture
from pokemon_agent.config import LearnerConfig
from pokemon_agent.files import atomic_json
from pokemon_agent.model.serialization import load_weights, model_config, read_checkpoint
from pokemon_agent.model.transformer import make_model


class CheckpointTests(unittest.TestCase):
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

    def dataset(self):
        return fixture_dataset(self.model, self.obs, self.actions)

    def test_failed_json_flush_preserves_previous_status(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"status.json"
            atomic_json(path, dict(status="previous"))
            with patch("pokemon_agent.files.os.fsync", side_effect=OSError("disk flush failed")):
                with self.assertRaisesRegex(OSError, "disk flush failed"):
                    atomic_json(path, dict(status="new"))
            self.assertEqual(json.loads(path.read_text()), dict(status="previous"))

    def test_resume_restores_counters_optimizer_rng_and_pending_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = bare_learner(temporary, self.architecture, self.cfg)
            source.dataset = self.dataset()
            source.total_steps, source.total_episodes, source.update_idx = 123, 4, 7
            saved = source._save_checkpoint(Path(temporary)/"learner_update_000000007.pt")
            expected_random = torch.rand(3)
            restored = bare_learner(temporary, self.architecture, self.cfg)
            restored._load_checkpoint(saved)
            self.assertEqual((restored.total_steps, restored.total_episodes, restored.update_idx), (123, 4, 7))
            self.assertEqual(len(restored.dataset), len(source.dataset))
            torch.testing.assert_close(expected_random, torch.rand(3))
            for key, value in source.net.state_dict().items():
                torch.testing.assert_close(value, restored.net.state_dict()[key])

    def test_phase_change_does_not_import_imitation_step_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = bare_learner(temporary, self.architecture, self.cfg, mode="imitation")
            source.total_steps, source.update_idx = 50000, 15
            saved = source._save_checkpoint(Path(temporary)/"imitation.pt")
            ppo = bare_learner(temporary, self.architecture, self.cfg)
            ppo._load_checkpoint(saved)
            self.assertEqual((ppo.total_steps, ppo.update_idx), (0, 0))

    def test_warmup_to_ppo_starts_with_fresh_optimizer_and_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            warmup = bare_learner(temporary, self.architecture, self.cfg, mode="warmup")
            warmup.total_steps, warmup.update_idx = 20792, 20
            warmup.dataset = self.dataset()
            saved = warmup._save_checkpoint(Path(temporary)/"warmup.pt")
            ppo = bare_learner(temporary, self.architecture, self.cfg)
            ppo._load_checkpoint(saved)
            self.assertEqual((ppo.total_steps, ppo.update_idx, ppo.total_episodes), (0, 0, 0))
            self.assertEqual(len(ppo.dataset), 0)
            self.assertEqual(len(ppo.opt.state), 0)

    def test_resume_refuses_settings_it_would_ignore(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = bare_learner(temporary, self.architecture, ppo_lr_multipliers="current")._save_checkpoint(
                Path(temporary)/"learner_update_000000001.pt")
            for changed in (dict(ppo_lr_multipliers="historical"), dict(ppo_lr_multipliers="current", lr=3e-5),
                            dict(ppo_lr_multipliers="current", lr_schedule="constant")):
                with self.assertRaisesRegex(ValueError, "optimizer settings"):
                    bare_learner(temporary, self.architecture, **changed)._load_checkpoint(saved)
            bare_learner(temporary, self.architecture, ppo_lr_multipliers="current")._load_checkpoint(saved)
            # Older checkpoints predate the setting and used the current multipliers.
            checkpoint = torch.load(saved, weights_only=False)
            del checkpoint["run_cfg"]["learner"]["ppo_lr_multipliers"]
            legacy = Path(temporary)/"legacy.pt"
            torch.save(checkpoint, legacy)
            with self.assertRaisesRegex(ValueError, "ppo_lr_multipliers"):
                bare_learner(temporary, self.architecture)._load_checkpoint(legacy)
            bare_learner(temporary, self.architecture, ppo_lr_multipliers="current")._load_checkpoint(legacy)

    def test_milestones_are_saved_once_and_load_for_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            learner = bare_learner(temporary, self.architecture, milestone_every_states=100)
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
            self.assertIsNone(bare_learner(temporary, self.architecture, milestone_every_states=0)._maybe_save_milestone())


if __name__ == "__main__":
    unittest.main()
