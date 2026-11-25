# pokemon-agent

A 15-token Transformer policy for Pokémon Showdown `gen9randombattle`. It first
imitates poke-env's `SimpleHeuristicsPlayer`, then improves by PPO self-play
across many simultaneous battles, using Ray. The full record of the current run,
its results and its exact commands is in
[docs/reproduction-log.md](docs/reproduction-log.md).

## Install

Python 3.12 or newer. Install a CUDA build of PyTorch 2.7.1 first if you have
an NVIDIA GPU, then install this package in editable mode:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install -e .
```

`requirements-lock.txt` pins every package of the tested environment.

Training also needs a built Pokémon Showdown checkout with the bundled
`showdown-plugin/rlspawn.ts` copied into `server/chat-plugins/` (see
[the log](docs/reproduction-log.md#new-showdown-checkout)). Collection and
evaluation only need ordinary challenges.

## Commands

Run them from the repository root. Logs, checkpoints and data are written under
the current folder (`logs/`, `checkpoints/`, `data/`), or under
`POKEMON_AGENT_HOME` when that variable is set.

| Command | What it does | Was |
| --- | --- | --- |
| `pokemon-agent-local` | Starts its own Showdown server(s), runs `--entry train`, `collect` or `evaluate`, then stops them | `run_local.py` |
| `pokemon-agent-train` | Distributed imitation, critic warmup or PPO self-play against servers you started | `train.py` |
| `pokemon-agent-collect` | Records both sides of heuristic-vs-heuristic battles into shards | `collect.py` |
| `pokemon-agent-imitate` | Offline imitation over collected shards, with held-out scoring | `imitation.py` |
| `pokemon-agent-evaluate` | Plays a fixed checkpoint against the corrected or stock heuristic | `evaluate.py` |
| `pokemon-agent-upgrade` | Adds move-category inputs to an older checkpoint without changing its outputs | `upgrade_checkpoint.py` |

Each also runs as a module, for example `python -m pokemon_agent.cli.train --help`.
To stop training cleanly, create `logs/stop-training` (or run
`scripts/Stop-LocalTraining.ps1`); the driver saves a checkpoint and exits.

A two-update CPU smoke run:

```bash
pokemon-agent-local --server-dir path/to/showdown --device cpu --mode ppo \
  --model-config tests/fixtures/tiny_model.json --no-resume --steps-per-update 64 \
  --batch-size 16 --microbatch-size 8 --epochs 1 --max-updates 2 \
  --checkpoint-dir checkpoints/smoke --save-every 1
```

## Layout

```
src/pokemon_agent/
├── paths.py  files.py  config.py   where outputs go, atomic writes, all settings
├── assets/vocab.json               entity vocabulary for the observation
├── vendor/                         upstream de8cc69 sources, byte-for-byte
├── obs/        battle → 8,567 numbers (layout, actions, per-block encoders)
├── model/      the 15-token network, observation unpacker, checkpoint loading
├── rl/         masked sampling, value targets, GAE, rewards, buffer, PPO update
├── training/   optimizer, checkpoints, datasets, metrics, offline imitation
├── env/        poke-env players, patches, Showdown helpers, the thread bridge
├── distributed/  Ray actors: weights, inference, learner, rollout workers, launch
└── cli/        the six commands above
```

Each layer imports only the ones above it in this list: `obs` uses NumPy and
poke-env only through `obs/poke_env_compat.py`; `model`, `rl` and `training`
need PyTorch but neither poke-env nor Ray; only `distributed` and `cli/train.py`
need Ray. Importing the package changes nothing in poke-env; each command calls
`pokemon_agent.env.patches.apply_all()` at start-up.

## Tests

```bash
python -m unittest discover -s tests
```

47 tests run in a few seconds on CPU. One is skipped unless the 10k-battle
seed checkpoint is present at `checkpoints/seed/imitation-best.pt`.

## License

MIT; see `LICENSE`. The training code started from the MIT-licensed ps-ppo
project, whose notice is in `LICENSE-ps-ppo`.
