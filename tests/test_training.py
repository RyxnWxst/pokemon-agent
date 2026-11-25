"""Verify policy compatibility, actual PPO updates, freezing, and checkpoint resume."""
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import torch

from checkpoint_io import atomic_json, load_weights, model_config, read_checkpoint
from config import ROOT, LearnerConfig, ModelConfig, RunConfig
from learner import LearnerActor
from metrics import action_metrics
from model import make_model
from obs_assembler import ObservationAssembler
from optimizers import make_optimizer
from ppo_core import AsyncEpisodeDataset, gae_from_episode, masked_logprob_entropy, masked_sample, ppo_update

FIXTURES = ROOT/"tests/fixtures"


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.fixture = torch.load(FIXTURES/"battle-observations.pt", weights_only=True)
        cls.architecture = ModelConfig(**json.loads((ROOT/"tests/tiny_model.json").read_text()))

    def setUp(self):
        torch.manual_seed(71)
        self.model = make_model(self.architecture).eval()
        self.obs = self.fixture["observations"]
        self.actions = self.fixture["actions"]
        self.cfg = replace(LearnerConfig(), device="cpu", mode="ppo", lr=.001, lr_warmup_steps=0,
            update_epochs=1, minibatch_size=3, target_kl=0.)

    def dataset(self):
        with torch.no_grad():
            logits, _, values, *_ = self.model(self.obs)
            start, end = self.model.unpacker.offsets["action_mask"]
            logp, _ = masked_logprob_entropy(logits, self.obs[:, start:end], self.actions)
        n = len(self.actions)
        dataset = AsyncEpisodeDataset(14, "cpu")
        dataset.add_steps(self.obs, self.actions, logp, values, torch.linspace(-1, 1, n),
            torch.linspace(1, -1, n), torch.zeros(n), torch.arange(n))
        return dataset

    def test_fifteen_tokens_and_independent_actor(self):
        self.assertEqual(self.model.total_tokens, 15)
        self.assertEqual(tuple(self.model.attn_mask.shape), (15, 15))
        before = self.model(self.obs)[0].detach()
        with torch.no_grad():
            self.model.critic_tok.add_(100)
        torch.testing.assert_close(before, self.model(self.obs)[0], atol=1e-6, rtol=1e-6)
        together = self.model(self.obs)[0]
        separate = torch.cat([self.model(row[None])[0] for row in self.obs])
        torch.testing.assert_close(together, separate, atol=1e-5, rtol=1e-5)

    def test_ppo_changes_policy_and_value_weights(self):
        dataset = self.dataset()
        policy = self.model.pi_head.weight.detach().clone()
        value = self.model.v_head.weight.detach().clone()
        optimizer, scheduler = make_optimizer(self.model, self.cfg)
        stats = ppo_update(self.model, optimizer, dataset, scheduler=scheduler,
            mode="ppo", **self.cfg.ppo_kwargs(), v_min=-1.6, v_max=1.6, v_bins=51)
        self.assertEqual(stats.n_mb, 3)  # Includes final two decisions.
        self.assertTrue(torch.isfinite(torch.tensor(stats.total_loss)))
        self.assertFalse(torch.equal(policy, self.model.pi_head.weight))
        self.assertFalse(torch.equal(value, self.model.v_head.weight))

    def test_imitation_freezes_critic_and_rejects_illegal_labels(self):
        cfg = replace(self.cfg, mode="imitation")
        optimizer, _ = make_optimizer(self.model, cfg)
        critic = self.model.v_head.weight.detach().clone()
        self.assertFalse(self.model.critic_tok.requires_grad)
        ppo_update(self.model, optimizer, self.dataset(), mode="imitation", **cfg.ppo_kwargs())
        torch.testing.assert_close(critic, self.model.v_head.weight, atol=0, rtol=0)
        dataset = self.dataset()
        start, end = self.model.unpacker.offsets["action_mask"]
        dataset.obs[0] = dataset.obs[0].clone()
        dataset.obs[0][:, start:end] = 0
        with self.assertRaisesRegex(ValueError, "legal action mask"):
            ppo_update(self.model, optimizer, dataset, mode="imitation", **cfg.ppo_kwargs())

    def test_critic_warmup_preserves_policy(self):
        before = self.model(self.obs)[0].detach()
        optimizer, _ = make_optimizer(self.model, replace(self.cfg, mode="warmup"))
        ppo_update(self.model, optimizer, self.dataset(), mode="warmup", **self.cfg.ppo_kwargs())
        self.model.eval()
        torch.testing.assert_close(before, self.model(self.obs)[0], atol=1e-6, rtol=1e-6)

    def test_metrics_score_every_decision(self):
        metrics = action_metrics(self.model, self.dataset(), batch_size=3)
        self.assertEqual(metrics["decisions"], len(self.actions))
        self.assertEqual(sum(item["decisions"] for item in metrics["action_groups"].values()), len(self.actions))

    def test_sampling_log_probability_matches_distribution(self):
        logits = torch.tensor([[1., -2., 3.]])
        mask = torch.tensor([[1., 0., 1.]])
        actions, logp, _ = masked_sample(logits, mask, temp=.5)
        expected, _ = masked_logprob_entropy(logits/.5, mask, actions)
        torch.testing.assert_close(logp, expected)

    def test_failed_json_flush_preserves_previous_status(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"status.json"
            atomic_json(path, dict(status="previous"))
            with patch("checkpoint_io.os.fsync", side_effect=OSError("disk flush failed")):
                with self.assertRaisesRegex(OSError, "disk flush failed"):
                    atomic_json(path, dict(status="new"))
            self.assertEqual(json.loads(path.read_text()), dict(status="previous"))

    def test_terminal_returns_do_not_bootstrap(self):
        advantages, returns = gae_from_episode(torch.tensor([0., 1.]), torch.tensor([.2, .3]),
            torch.tensor([0., 1.]), gamma=1., lam=1.)
        torch.testing.assert_close(returns, torch.ones(2))

    def test_current_kl_rule_can_stop_during_first_pass(self):
        dataset = self.dataset()
        dataset.logp[0] = dataset.logp[0] + 1.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.)
        stats = ppo_update(self.model, optimizer, dataset, update_epochs=3, minibatch_size=3,
            target_kl=.02, kl_stop_rule="minibatch")
        self.assertTrue(stats.early_stopped)
        self.assertEqual(stats.n_mb, 1)
        self.assertEqual(stats.first_pass_samples, 3)
        self.assertEqual(stats.epochs_completed, 0)

    def test_historical_kl_rule_completes_first_pass(self):
        dataset = self.dataset()
        dataset.logp[0] = dataset.logp[0] + 1.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.)
        stats = ppo_update(self.model, optimizer, dataset, update_epochs=3, minibatch_size=3,
            target_kl=.02, kl_stop_rule="historical", microbatch_size=2)
        self.assertTrue(stats.early_stopped)
        self.assertEqual(stats.n_mb, 4)
        self.assertEqual(stats.first_pass_samples, len(dataset))
        self.assertEqual(stats.epochs_completed, 1)

    def test_gradient_accumulation_matches_full_batches_and_partial_tail(self):
        import copy
        dataset = self.dataset()
        accumulated = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.001)
        micro_optimizer = torch.optim.SGD(accumulated.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        micro_scheduler = torch.optim.lr_scheduler.StepLR(micro_optimizer, step_size=1, gamma=.9)
        torch.manual_seed(123)
        reference = ppo_update(self.model, optimizer, dataset, scheduler=scheduler,
            update_epochs=2, minibatch_size=5, target_kl=0., microbatch_size=0)
        torch.manual_seed(123)
        micro = ppo_update(accumulated, micro_optimizer, dataset, scheduler=micro_scheduler,
            update_epochs=2, minibatch_size=5, target_kl=0., microbatch_size=2)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, accumulated.state_dict()[key], atol=1e-6, rtol=1e-5)
        self.assertEqual(reference.n_mb, micro.n_mb)
        self.assertEqual(scheduler.last_epoch, micro_scheduler.last_epoch)
        self.assertEqual(micro.samples_processed, 2*len(dataset))
        self.assertEqual(micro.first_pass_samples, len(dataset))
        self.assertEqual(micro.epochs_completed, 2)
        self.assertGreater(micro.n_microbatches, reference.n_microbatches)
        self.assertAlmostEqual(reference.total_loss, micro.total_loss, places=5)

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

    def bare_learner(self, directory, mode="ppo"):
        learner = LearnerActor.__new__(LearnerActor)
        learner.cfg = replace(self.cfg, mode=mode, ckpt_dir=str(directory))
        learner.run_cfg = replace(RunConfig.default(), model=self.architecture, learner=learner.cfg)
        learner.net = make_model(self.architecture)
        learner.opt, learner.sched = make_optimizer(learner.net, learner.cfg)
        learner.dataset = AsyncEpisodeDataset(14, "cpu")
        learner.update_idx = learner.total_steps = learner.total_episodes = 0
        learner.directory = Path(directory)
        return learner

    def test_resume_restores_counters_optimizer_rng_and_pending_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.bare_learner(temporary)
            source.dataset = self.dataset()
            source.total_steps, source.total_episodes, source.update_idx = 123, 4, 7
            saved = source._save_checkpoint(Path(temporary)/"learner_update_000000007.pt")
            expected_random = torch.rand(3)
            restored = self.bare_learner(temporary)
            restored._load_checkpoint(saved)
            self.assertEqual((restored.total_steps, restored.total_episodes, restored.update_idx), (123, 4, 7))
            self.assertEqual(len(restored.dataset), len(source.dataset))
            torch.testing.assert_close(expected_random, torch.rand(3))
            for key, value in source.net.state_dict().items():
                torch.testing.assert_close(value, restored.net.state_dict()[key])

    def test_phase_change_does_not_import_imitation_step_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.bare_learner(temporary, mode="imitation")
            source.total_steps, source.update_idx = 50000, 15
            saved = source._save_checkpoint(Path(temporary)/"imitation.pt")
            ppo = self.bare_learner(temporary)
            ppo._load_checkpoint(saved)
            self.assertEqual((ppo.total_steps, ppo.update_idx), (0, 0))

    def test_warmup_to_ppo_starts_with_fresh_optimizer_and_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            warmup = self.bare_learner(temporary, mode="warmup")
            warmup.total_steps, warmup.update_idx = 20792, 20
            warmup.dataset = self.dataset()
            saved = warmup._save_checkpoint(Path(temporary)/"warmup.pt")
            ppo = self.bare_learner(temporary)
            ppo._load_checkpoint(saved)
            self.assertEqual((ppo.total_steps, ppo.update_idx, ppo.total_episodes), (0, 0, 0))
            self.assertEqual(len(ppo.dataset), 0)
            self.assertEqual(len(ppo.opt.state), 0)

    def test_imports_are_self_contained(self):
        for name in ("config", "model", "ppo_core", "obs_assembler", "obs_moves", "obs_pokemon",
                     "obs_global", "obs_abilities", "obs_transitions", "inference", "learner", "worker",
                     "evaluate", "collect", "imitation", "checkpoint_io", "optimizers", "utils"):
            self.assertTrue(Path(importlib.import_module(name).__file__).resolve().is_relative_to(ROOT), name)
        self.assertTrue(ObservationAssembler().total_dim > 0)

    def test_historical_source_hashes_match_provenance(self):
        manifest = json.loads((ROOT/"vendor/source_manifest.json").read_text())
        for source in manifest["sources"].values():
            digest = hashlib.sha256((ROOT/"vendor"/source["copy"]).read_bytes()).hexdigest()
            self.assertEqual(digest, source["sha256"])


class SavedPolicyTests(unittest.TestCase):
    def test_saved_69_percent_policy_matches_original_outputs(self):
        torch.set_num_threads(2)
        path = ROOT/"checkpoints/seed/imitation-best.pt"
        if not path.exists():
            self.skipTest("Optional pretrained checkpoint not bundled")
        checkpoint = read_checkpoint(path)
        model = make_model(model_config(checkpoint)).eval()
        load_weights(model, checkpoint)
        fixture = torch.load(FIXTURES/"battle-observations.pt", weights_only=True)
        reference = torch.load(FIXTURES/"reference-policy.pt", weights_only=True)
        with torch.no_grad():
            logits, value_logits, values, *_ = model(fixture["observations"])
        torch.testing.assert_close(logits, reference["logits"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(value_logits, reference["value_logits"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(values, reference["values"], atol=1e-6, rtol=1e-6)
        self.assertEqual(model.total_tokens, 15)


if __name__ == "__main__":
    unittest.main()
