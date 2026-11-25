"""Per-step rewards for a finished trajectory: faint shaping plus the terminal result.

Pure NumPy, so the reward can be tested without Ray, poke-env or a server.
"""
import numpy as np


def faint_counts(observations: np.ndarray, meta: dict):
    """Fainted Pokémon at each step, (yours, the opponent's), read from the observations' fainted bits."""
    b_start, b_end = meta["offsets"]["pokemon_body"]
    steps = len(observations)
    body_history = observations[:, b_start:b_end].reshape(steps, 12, meta["dim_pokemon_body"])
    is_fainted = body_history[:, :, meta["faint_internal_idx"]] > 0.5
    return is_fainted[:, :6].sum(axis=1), is_fainted[:, 6:].sum(axis=1)


def trajectory_rewards(observations: np.ndarray, won: bool, lost: bool, reward_cfg, meta: dict):
    """Rewards and done flags for one player's trajectory.

    A faint seen between observations t and t+1 is credited to step t (+faint_opp
    for each opponent Pokémon, faint_self for each of yours); a faint after the
    last decision earns nothing beyond the terminal reward on the last step.
    """
    steps = len(observations)
    terminal_reward = (reward_cfg.terminal_win if won else
        reward_cfg.terminal_loss if lost else 0.)
    rewards = np.zeros(steps, dtype=np.float32)

    if reward_cfg.use_faint_reward:
        own, opponent = faint_counts(observations, meta)
        ds = np.diff(own)
        do = np.diff(opponent)
        # Apply to rewards[:-1] to align the consequence with the action
        rewards[:-1] += (np.maximum(0, ds) * float(reward_cfg.faint_self)) + \
                        (np.maximum(0, do) * float(reward_cfg.faint_opp))

    rewards[-1] += terminal_reward
    dones = np.zeros(steps, dtype=np.float32)
    dones[-1] = 1.0
    return rewards, dones
