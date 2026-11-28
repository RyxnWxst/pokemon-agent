"""Per-step rewards for one player's trajectory.

Three parts, all computed from the player's own observations:

* Terminal: +1 win, -1 loss, 0 tie, on the last decision.
* Faints: -0.1 for each of our Pokemon that faints between two decisions and
  +0.1 for each opponent Pokemon. ``faint_index="legacy"`` reproduces runs made
  before the index fix, which read the "ability unknown" flag instead.
* HP-lead shaping (optional, ``hp_shaping`` = c > 0): potential-based shaping
  with Phi(s) = c * (our mean HP fraction - opponent mean HP fraction). Each step
  gets gamma * Phi(next) - Phi(this), and Phi is 0 once the game is over. Opponent
  Pokemon not yet seen count as full HP, so a reveal is not a penalty. Both teams
  start at full HP (Phi = 0), so the shaping telescopes to (almost exactly) zero
  over a game: it moves credit earlier without changing which policy is best.
"""
import numpy as np

TEAM_SIZE = 6


def pokemon_views(obs, offsets, meta):
    """Return per-slot body features [T, 12, D] and species ids [T, 12]."""
    obs = np.asarray(obs, dtype=np.float32)
    steps = len(obs)
    start, end = offsets["pokemon_body"]
    body = obs[:, start:end].reshape(steps, 2*TEAM_SIZE, meta["dim_pokemon_body"])
    start, end = offsets["pokemon_ids"]
    ids = obs[:, start:end].reshape(steps, 2*TEAM_SIZE, -1)[:, :, 0]
    return body, ids


def hp_lead(body, ids, hp_index=0):
    """Our mean HP fraction minus the opponent's; unseen Pokemon (species id 0) count as full HP."""
    hp = body[:, :, hp_index] / 100.
    hp = np.where(ids == 0, 1., hp)
    return hp[:, :TEAM_SIZE].mean(axis=1) - hp[:, TEAM_SIZE:].mean(axis=1)


def episode_rewards(obs, won, lost, reward_cfg, gamma, offsets, meta):
    """Rewards aligned with decisions: reward[t] follows the action taken at decision t."""
    steps = len(obs)
    rewards = np.zeros(steps, dtype=np.float32)
    if steps == 0:
        return rewards
    needs_views = reward_cfg.use_faint_reward or reward_cfg.hp_shaping
    body, ids = pokemon_views(obs, offsets, meta) if needs_views else (None, None)
    if reward_cfg.use_faint_reward:
        index = meta["faint_internal_idx" if reward_cfg.faint_index == "fixed" else "legacy_faint_internal_idx"]
        fainted = body[:, :, index] > .5
        ours = np.diff(fainted[:, :TEAM_SIZE].sum(axis=1))
        theirs = np.diff(fainted[:, TEAM_SIZE:].sum(axis=1))
        rewards[:-1] += (np.maximum(0, ours)*float(reward_cfg.faint_self)
                         + np.maximum(0, theirs)*float(reward_cfg.faint_opp))
    if reward_cfg.hp_shaping:
        potential = float(reward_cfg.hp_shaping) * hp_lead(body, ids, meta["feature_map"]["body"]["hp_int"])
        following = np.append(potential[1:], 0.)  # Phi is zero after the game ends.
        rewards += (gamma*following - potential).astype(np.float32)
    rewards[-1] += reward_cfg.terminal_win if won else reward_cfg.terminal_loss if lost else 0.
    return rewards
