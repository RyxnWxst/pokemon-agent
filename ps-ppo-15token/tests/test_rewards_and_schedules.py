"""Faint-index fix, potential-based HP shaping, heuristic opponent and the linear LR schedule."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from config import ROOT, LearnerConfig, ModelConfig, RewardConfig
from model import make_model
from obs_assembler import ObservationAssembler
from optimizers import make_optimizer
from rewards import episode_rewards, hp_lead, pokemon_views


class Battle:
    """Builds per-decision observations with only the fields the rewards read."""
    def __init__(self, steps):
        self.assembler = ObservationAssembler()
        self.meta, self.offsets = self.assembler.meta, self.assembler.offsets
        self.obs = np.zeros((steps, self.offsets["action_mask"][1]), dtype=np.float32)
        for slot in range(6):  # Our whole team is known; the opponent's lead is revealed.
            self.set(slice(None), slot, species=10+slot, hp=100)
        self.set(slice(None), 6, species=50, hp=100)

    def set(self, steps, slot, species=None, hp=None, fainted=None, ability_unknown=None):
        start = self.offsets["pokemon_body"][0] + slot*self.meta["dim_pokemon_body"]
        if species is not None:
            self.obs[steps, self.offsets["pokemon_ids"][0] + 2*slot] = species
        if hp is not None:
            self.obs[steps, start] = hp
        if fainted is not None:
            self.obs[steps, start + self.meta["faint_internal_idx"]] = fainted
        if ability_unknown is not None:
            self.obs[steps, start + self.meta["legacy_faint_internal_idx"]] = ability_unknown

    def rewards(self, won=False, lost=False, gamma=1., **reward):
        return episode_rewards(self.obs, won, lost, RewardConfig(**reward), gamma, self.offsets, self.meta)


class FaintIndexTests(unittest.TestCase):
    def test_fixed_index_is_the_fainted_flag(self):
        meta = ObservationAssembler().meta
        body = meta["feature_map"]["body"]
        self.assertEqual(meta["faint_internal_idx"], body["flags_raw"][0] + 1)  # active, fainted, tera, ...
        self.assertEqual(meta["legacy_faint_internal_idx"], body["mechanics_raw"][0] + 1)  # "ability unknown"
        self.assertEqual((meta["faint_internal_idx"], meta["legacy_faint_internal_idx"]), (113, 125))

    def test_recorded_fainted_pokemon_have_no_hp(self):
        assembler = ObservationAssembler()
        fixture = torch.load(ROOT/"tests/fixtures/battle-observations.pt", weights_only=True)
        body, _ = pokemon_views(fixture["observations"].numpy(), assembler.offsets, assembler.meta)
        fainted = body[:, :, assembler.meta["faint_internal_idx"]] > .5
        self.assertGreater(int(fainted.sum()), 0)
        self.assertTrue((body[:, :, 0][fainted] == 0).all())

    def test_faints_are_rewarded_on_the_preceding_decision(self):
        battle = Battle(4)
        battle.set(slice(2, None), 0, hp=0, fainted=1)     # we lose a Pokemon between decisions 1 and 2
        battle.set(slice(3, None), 6, hp=0, fainted=1)     # they lose one between decisions 2 and 3
        np.testing.assert_allclose(battle.rewards(won=True), [0, -.1, .1, 1.], atol=1e-6)

    def test_legacy_index_reproduces_the_old_reward(self):
        battle = Battle(4)
        battle.set(slice(2, None), 0, hp=0, fainted=1)
        battle.set(slice(None), 6, ability_unknown=1)
        battle.set(slice(2, None), 7, species=51, hp=100, ability_unknown=1)  # a new opponent Pokemon appears
        np.testing.assert_allclose(battle.rewards(lost=True, faint_index="legacy"), [0, .1, 0, -1.], atol=1e-6)
        np.testing.assert_allclose(battle.rewards(lost=True), [0, -.1, 0, -1.], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "faint_index"):
            RewardConfig(faint_index="other")


class HpShapingTests(unittest.TestCase):
    def test_unseen_opponents_count_as_full_hp(self):
        battle = Battle(3)
        battle.set(slice(1, None), 7, species=51, hp=100)  # a full-HP reveal is not a change in HP lead
        body, ids = pokemon_views(battle.obs, battle.offsets, battle.meta)
        np.testing.assert_allclose(hp_lead(body, ids), [0, 0, 0])
        np.testing.assert_allclose(battle.rewards(use_faint_reward=False, hp_shaping=.5), [0, 0, 0], atol=1e-6)

    def test_shaping_follows_hp_swings_and_sums_to_zero(self):
        battle = Battle(4)
        battle.set(slice(1, None), 6, hp=40)   # we take 60% off their lead
        battle.set(slice(2, None), 0, hp=70)   # they take 30% off ours
        rewards = battle.rewards(won=True, use_faint_reward=False, hp_shaping=.6)
        lead = np.array([0, .6, .3, .3]) / 6
        expected = .6*np.append(np.diff(lead), -lead[-1])
        expected[-1] += 1.
        np.testing.assert_allclose(rewards, expected, atol=1e-6)
        # With gamma 1 the shaping telescopes from Phi(start) = 0 to Phi(end) = 0.
        self.assertAlmostEqual(float(rewards.sum()), 1., places=6)

    def test_disabled_shaping_changes_nothing(self):
        battle = Battle(3)
        battle.set(slice(1, None), 6, hp=10)
        np.testing.assert_allclose(battle.rewards(won=True), [0, 0, 1.], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "hp_shaping"):
            RewardConfig(hp_shaping=-1)


class OpponentAndScheduleTests(unittest.TestCase):
    def test_heuristic_opponent_settings(self):
        self.assertFalse(LearnerConfig().heuristic_opponent)
        self.assertTrue(LearnerConfig(mode="ppo_vs_teacher").heuristic_opponent)
        self.assertTrue(LearnerConfig(mode="warmup", opponent="heuristic").heuristic_opponent)
        with self.assertRaisesRegex(ValueError, "Imitation"):
            LearnerConfig(mode="imitation", opponent="heuristic")

    def test_linear_schedule_follows_new_states(self):
        model = make_model(ModelConfig(**json.loads((ROOT/"tests/tiny_model.json").read_text())))
        done = [0.]
        cfg = replace(LearnerConfig(), device="cpu", mode="ppo", lr_schedule="linear", max_total_steps=1000,
            lr_warmup_steps=2, lr_final_fraction=.1)
        optimizer, scheduler = make_optimizer(model, cfg, progress=lambda: done[0])
        rates = []
        for fraction in (0., 0., .5, 1., 1.):
            done[0] = fraction
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"] / cfg.lr)
        np.testing.assert_allclose(rates, [1., 1., .55, .1, .1], atol=1e-9)
        with self.assertRaisesRegex(ValueError, "state target"):
            LearnerConfig(lr_schedule="linear")

    def test_cli_options(self):
        from train import parse_args
        args = parse_args([])
        self.assertEqual((args.faint_reward, args.hp_shaping, args.opponent, args.lr_schedule),
                         ("fixed", 0., "self", "historical"))
        args = parse_args(["--mode", "warmup", "--opponent", "heuristic", "--hp-shaping", ".5",
                           "--lr-schedule", "linear", "--total-steps", "100", "--lr-warmup-steps", "100"])
        self.assertEqual((args.opponent, args.hp_shaping, args.lr_warmup_steps), ("heuristic", .5, 100))
        with patch("sys.stderr"):
            for invalid in (["--lr-schedule", "linear"], ["--mode", "imitation", "--opponent", "heuristic"],
                            ["--hp-shaping", "-1"], ["--lr-final-fraction", "2", "--total-steps", "5"]):
                with self.assertRaises(SystemExit):
                    parse_args(invalid)


class ExperimentPlanTests(unittest.TestCase):
    def test_every_stage_builds_valid_training_arguments(self):
        from experiments import DEFAULT_PLAN, ppo_args, resolve_mode, warmup_args
        from train import parse_args
        for stage in DEFAULT_PLAN["stages"]:
            for best_mode in ("ppo", "ppo_vs_teacher"):
                mode = resolve_mode(stage["mode"], best_mode)
                warm = parse_args(DEFAULT_PLAN["train_args"] + warmup_args(stage, mode, "start.pt", "warm"))
                self.assertEqual((warm.mode, warm.max_updates, warm.hp_shaping), ("warmup", 10, .5))
                self.assertEqual(warm.opponent, "heuristic" if mode == "ppo_vs_teacher" else "self")
                run = parse_args(DEFAULT_PLAN["train_args"] + ppo_args(stage, mode, "w.pt", "ppo", 5_000_000))
                self.assertEqual((run.mode, run.total_steps, run.faint_reward), (mode, stage["states"], "fixed"))
        self.assertEqual(resolve_mode("auto", "ppo_vs_teacher"), "ppo_vs_teacher")
        self.assertEqual(resolve_mode("auto", None), "ppo")

    def test_a_milestone_is_promoted_only_when_its_pooled_result_is_better(self):
        import shutil
        import experiments
        workspace = "unittest-pipeline"
        pipeline = experiments.Pipeline(workspace=workspace)
        try:
            results = {}

            def fake_evaluate(checkpoint, battles, seed, label):
                wins = results[(Path(checkpoint).name, seed)]
                record = dict(checkpoint=experiments.relative(checkpoint), label=label, seed=seed,
                              battles=battles, wins=wins, win_rate=wins/battles)
                pipeline.evaluations.append(record)
                return record

            pipeline.evaluate = fake_evaluate
            baseline = experiments.RUNS/"baseline.pt"
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_bytes(b"baseline")
            results[("baseline.pt", 71)] = 1720               # 86.0% over 2,000
            pipeline.evaluate(baseline, 2000, 71, "baseline")
            pipeline.set_best(baseline, "baseline", "ppo", copy=None)
            cases = [("worse.pt", 850, None, False),           # 85.0% on the first test: no confirmation
                     ("lucky.pt", 880, 840, False),            # 88.0%, then pooled 86.0%: not better
                     ("better.pt", 890, 880, True)]            # pooled 88.5%: promoted
            for name, first, confirm, promoted in cases:
                milestone = experiments.RUNS/name
                milestone.write_bytes(name.encode())
                results[(name, 71)], results[(name, 72)] = first, confirm
                pipeline.consider(milestone, "stage", "ppo_vs_teacher")
                self.assertEqual(pipeline.best["checkpoint"].endswith(name), promoted, name)
            self.assertEqual((experiments.BEST/"best.pt").read_bytes(), b"better.pt")
            self.assertEqual((pipeline.best["wins"], pipeline.best["battles"], pipeline.best["mode"]),
                             (1770, 2000, "ppo_vs_teacher"))
            self.assertEqual([entry["stage"] for entry in pipeline.best["history"]], ["baseline", "stage"])
        finally:
            for directory in (experiments.LOGS, experiments.RUNS, experiments.BEST):
                shutil.rmtree(directory, ignore_errors=True)

    def test_wilson_interval_matches_evaluation_reports(self):
        from experiments import wilson
        low, high = wilson(431, 500)  # logs/ppo-readme-eval-200m.json
        self.assertAlmostEqual(low, .829, places=3)
        self.assertAlmostEqual(high, .889, places=3)


if __name__ == "__main__":
    unittest.main()
