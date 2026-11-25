"""README-era move-category inputs: restoration, legacy loading, and output-preserving upgrade."""
from dataclasses import asdict, replace
import json
import unittest
import torch

from checkpoint_io import model_config
from config import ROOT, ModelConfig
from model import make_model, upgrade_move_category_inputs
from obs_assembler import ObservationAssembler
from vendor.historical_config import ModelConfig as HistoricalModelConfig

FIXTURES = ROOT/"tests/fixtures"


class MoveCategoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.obs = torch.load(FIXTURES/"battle-observations.pt", weights_only=True)["observations"].float()
        cls.architecture = ModelConfig(**json.loads((ROOT/"tests/tiny_model.json").read_text()))
        cls.legacy_architecture = replace(cls.architecture, move_category_inputs=False)
        meta = ObservationAssembler().meta
        start, end = meta["offsets"]["move_scalars"]
        cls.width = meta["dim_move_scalars"] // 4
        cls.span = (start, end)
        cls.category = meta["feature_map"]["move"]["category_raw"]

    def moves(self, observations):
        start, end = self.span
        return observations[:, start:end].reshape(len(observations), -1, self.width)

    def test_vocabulary_matches_historical_category_order(self):
        self.assertEqual(ObservationAssembler().vocab_lists["move.category"], ["physical", "special", "status"])

    def test_current_schema_leaves_historical_slots_empty(self):
        moves = self.moves(self.obs)
        self.assertEqual(float(moves[..., 3:6].abs().max()), 0.)
        self.assertGreater(int(moves[..., self.category[0]+1:self.category[1]].sum()), 0)

    def test_enabled_model_restores_one_category_per_revealed_move(self):
        torch.manual_seed(71)
        model = make_model(self.architecture)
        self.assertTrue(model.move_category_inputs)
        adapted = self.moves(model.adapt_observations(self.obs))
        source = self.moves(self.obs)[..., self.category[0]+1:self.category[1]]
        torch.testing.assert_close(adapted[..., 3:6], source, atol=0, rtol=0)
        revealed = source.sum(-1) > 0
        self.assertTrue(torch.equal(adapted[..., 3:6].sum(-1)[revealed], torch.ones(int(revealed.sum()))))
        self.assertEqual(int(revealed.sum()), 200)
        # Only the historical slots change; the turn rescaling is the other adaptation.
        legacy = make_model(self.legacy_architecture).adapt_observations(self.obs)
        changed = (model.adapt_observations(self.obs) != legacy).any(0).nonzero().flatten()
        self.assertTrue(set(changed.tolist()) <= set(model.category_destination.tolist()))

    def test_legacy_model_keeps_slots_empty(self):
        model = make_model(self.legacy_architecture)
        self.assertEqual(float(self.moves(model.adapt_observations(self.obs))[..., 3:6].abs().max()), 0.)

    def test_checkpoints_without_the_setting_load_as_legacy(self):
        legacy_config = asdict(HistoricalModelConfig())
        self.assertNotIn("move_category_inputs", legacy_config)
        self.assertFalse(model_config(dict(model_config=legacy_config)).move_category_inputs)
        self.assertTrue(model_config(dict(model_config=asdict(ModelConfig()))).move_category_inputs)
        self.assertTrue(ModelConfig().move_category_inputs)

    def test_upgrade_preserves_outputs_and_trains_category_columns(self):
        torch.manual_seed(71)
        legacy = make_model(self.legacy_architecture).eval()
        columns = legacy.category_weight_columns()
        with torch.no_grad():
            # Arbitrary values that a legacy model could never have used.
            legacy.move_net[0].weight[:, columns] = .5
            expected = [item.clone() for item in legacy(self.obs)[:3]]
        upgraded = upgrade_move_category_inputs(legacy).eval()
        self.assertTrue(upgraded.move_category_inputs)
        self.assertTrue(upgraded.reproduction_config["move_category_inputs"])
        with torch.no_grad():
            for before, after in zip(expected, upgraded(self.obs)[:3]):
                torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
        # Without zeroing, the same weights react to the restored inputs: the columns are live.
        flipped = make_model(self.architecture).eval()
        flipped.load_state_dict(legacy.state_dict(), strict=True)
        with torch.no_grad():
            self.assertGreater(float((flipped(self.obs)[0] - expected[0]).abs().max()), 1e-4)
        upgraded.train()
        upgraded(self.obs)[0].logsumexp(-1).sum().backward()
        self.assertGreater(float(upgraded.move_net[0].weight.grad[:, columns].abs().sum()), 0.)

    def test_upgrade_rejects_enabled_models(self):
        with self.assertRaisesRegex(ValueError, "already receives"):
            upgrade_move_category_inputs(make_model(self.architecture))


if __name__ == "__main__":
    unittest.main()
