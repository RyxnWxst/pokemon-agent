"""PPO, imitation and warmup updates: what each trains, sampling, GAE, KL stopping, minibatching."""
import copy
from dataclasses import replace
import unittest
import torch

from helpers import fixture_dataset, load_fixture, tiny_architecture
from pokemon_agent.config import LearnerConfig
from pokemon_agent.model.transformer import make_model
from pokemon_agent.rl.advantages import gae_from_episode
from pokemon_agent.rl.buffer import AsyncEpisodeDataset, minibatch_bounds
from pokemon_agent.rl.distributions import masked_logprob_entropy, masked_sample
from pokemon_agent.rl.ppo import ppo_update
from pokemon_agent.training.optimizer import make_optimizer


class PPOUpdateTests(unittest.TestCase):
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

    def test_sampling_log_probability_matches_distribution(self):
        logits = torch.tensor([[1., -2., 3.]])
        mask = torch.tensor([[1., 0., 1.]])
        actions, logp, _ = masked_sample(logits, mask, temp=.5)
        expected, _ = masked_logprob_entropy(logits/.5, mask, actions)
        torch.testing.assert_close(logp, expected)

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
        obs, actions = self.fixture["observations"], self.fixture["actions"]
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


if __name__ == "__main__":
    unittest.main()
