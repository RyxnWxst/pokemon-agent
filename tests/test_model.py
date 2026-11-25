"""The 15-token network: token count, actor independence, the saved seed policy, vendored sources."""
import hashlib
import json
import unittest
import torch

from helpers import FIXTURES, load_fixture, tiny_architecture
from pokemon_agent.model.serialization import load_model
from pokemon_agent.model.transformer import make_model
from pokemon_agent.paths import PACKAGE_DIR, default_seed_checkpoint


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.obs = load_fixture()["observations"]

    def test_fifteen_tokens_and_independent_actor(self):
        torch.manual_seed(71)
        model = make_model(tiny_architecture()).eval()
        self.assertEqual(model.total_tokens, 15)
        self.assertEqual(tuple(model.attn_mask.shape), (15, 15))
        before = model(self.obs)[0].detach()
        with torch.no_grad():
            model.critic_tok.add_(100)
        torch.testing.assert_close(before, model(self.obs)[0], atol=1e-6, rtol=1e-6)
        together = model(self.obs)[0]
        separate = torch.cat([model(row[None])[0] for row in self.obs])
        torch.testing.assert_close(together, separate, atol=1e-5, rtol=1e-5)

    def test_historical_source_hashes_match_provenance(self):
        vendor = PACKAGE_DIR/"vendor"
        manifest = json.loads((vendor/"source_manifest.json").read_text())
        for source in manifest["sources"].values():
            digest = hashlib.sha256((vendor/source["copy"]).read_bytes()).hexdigest()
            self.assertEqual(digest, source["sha256"])


class SavedPolicyTests(unittest.TestCase):
    def test_saved_69_percent_policy_matches_original_outputs(self):
        torch.set_num_threads(2)
        path = default_seed_checkpoint()
        if not path.exists():
            self.skipTest("Optional pretrained checkpoint not bundled")
        model = load_model(path)
        fixture = load_fixture()
        reference = torch.load(FIXTURES/"reference-policy.pt", weights_only=True)
        with torch.no_grad():
            logits, value_logits, values, *_ = model(fixture["observations"])
        torch.testing.assert_close(logits, reference["logits"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(value_logits, reference["value_logits"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(values, reference["values"], atol=1e-6, rtol=1e-6)
        self.assertEqual(model.total_tokens, 15)


if __name__ == "__main__":
    unittest.main()
