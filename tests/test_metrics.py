"""Held-out action metrics."""
from dataclasses import replace
import unittest
import torch

from helpers import fixture_dataset, load_fixture, tiny_architecture
from pokemon_agent.config import LearnerConfig
from pokemon_agent.model.transformer import make_model
from pokemon_agent.training.metrics import action_metrics


class MetricsTests(unittest.TestCase):
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

    def test_metrics_score_every_decision(self):
        metrics = action_metrics(self.model, self.dataset(), batch_size=3)
        self.assertEqual(metrics["decisions"], len(self.actions))
        self.assertEqual(sum(item["decisions"] for item in metrics["action_groups"].values()), len(self.actions))


if __name__ == "__main__":
    unittest.main()
