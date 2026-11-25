"""The observation layout is computed without poke-env, and matches what the assembler builds."""
import subprocess
import sys
import textwrap
import unittest

from pokemon_agent.obs.assembler import ObservationAssembler
from pokemon_agent.obs.layout import BLOCKS, block_offsets, observation_metadata


class LayoutTests(unittest.TestCase):
    def test_block_offsets_match_assembler(self):
        assembler = ObservationAssembler()
        meta = observation_metadata()
        layout, offsets, total = block_offsets(meta)
        self.assertEqual(offsets, assembler.offsets)
        self.assertEqual(meta["offsets"], assembler.offsets)
        self.assertEqual(total, assembler.total_dim)
        self.assertEqual(total, 8567)
        self.assertEqual([name for name, *_ in layout], list(BLOCKS))
        starts = [offsets[name][0] for name in BLOCKS]
        ends = [offsets[name][1] for name in BLOCKS]
        self.assertEqual(starts, [0] + ends[:-1])
        self.assertEqual(assembler.calc_buf.shape, (total,))

    def test_layout_and_model_import_without_poke_env(self):
        script = textwrap.dedent("""
            import importlib.abc, sys
            class Blocked(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in ("poke_env", "ray"):
                        raise ImportError(f"{name} must not be imported here")
            sys.meta_path.insert(0, Blocked())
            from pokemon_agent.obs.layout import observation_metadata
            from pokemon_agent.model.transformer import make_model
            from pokemon_agent.rl.ppo import ppo_update
            from pokemon_agent.rl.rewards import trajectory_rewards
            from pokemon_agent.training.checkpoints import save_learner_checkpoint
            assert observation_metadata()["offsets"]["action_mask"][1] == 8567
        """)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
