# pokemon-agent

A reinforcement-learning agent for Pokemon Showdown Gen 9 Random Battles. It
reproduces the result claimed by
[Nebraskinator/ps-ppo](https://github.com/Nebraskinator/ps-ppo): more than 85%
wins against poke-env's `SimpleHeuristicsPlayer`.

All code and full instructions are in [`ps-ppo-15token/`](ps-ppo-15token/README.md).

## Approach

1. **Imitation:** a 15-token Transformer policy (about 58M parameters) learns to
   copy the heuristic player from 50,000 recorded battles. It reaches 94.97%
   agreement with the heuristic's choices on held-out games.
2. **Self-play PPO:** the policy then trains against copies of itself, using
   upstream's README-era recipe on one RTX 3070.

## Results

500 battles per checkpoint against the corrected `SimpleHeuristicsPlayer`, with
actions sampled from the policy:

| Model | Win rate | 95% interval |
| --- | --- | --- |
| Imitation only | 42.0% | 37.8–46.4% |
| Self-play, 50M states | 75.0% | 71.0–78.6% |
| Self-play, 130M states | 87.2% | 84.0–89.8% |
| Self-play, 200M states | 86.2% | 82.9–88.9% |

A larger 2,000-battle test separated the two best checkpoints: **200M won
87.0%** (85.4–88.4%) and 130M won 84.9% (83.2–86.4%).

Experiments that continue from this model are in progress: a corrected faint
reward, HP-lead reward shaping, training against the heuristic, and
learning-rate decay. The first stage, self-play with the corrected reward and
HP shaping, finished with a best of 87.5% over 2,000 battles, a tie with the
200M model. The second, training against the heuristic, raised the best to
**91.9%** over 2,000 battles (90.6–93.0%). That stage trains against the same
opponent the test uses, so part of the gain may be specific to that opponent. See
[Experiments after 200M](ps-ppo-15token/README.md#experiments-after-200m).

Trained checkpoints are not stored in this repository; each is about 233 MB.
