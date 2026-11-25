# ps-ppo: 15-token reproduction

This project reproduces the upstream ps-ppo README result: a 15-token Transformer
policy is trained by imitating `SimpleHeuristicsPlayer`, then trained by PPO
self-play for about 250 million states. The upstream README reports more than
85% wins against that heuristic.

All Python code lives in this directory. The Python environment and a built
Pokemon Showdown server are external dependencies. So is the imitation dataset,
which lives in the parent project (see [Step 1](#step-1-imitation)).

## Results so far

The run is in progress (as of Nov 18, 2026). Each row is 500 battles against the
corrected heuristic, with actions sampled from the policy.

| Checkpoint | Wins | Win rate | 95% interval |
| --- | --- | --- | --- |
| Imitation seed | 210 | 42.0% | 37.8–46.4% |
| PPO update 75 (2.46M states) | 213 | 42.6% | 38.3–47.0% |
| 10M states | 286 | 57.2% | 52.8–61.5% |
| 20M states | 312 | 62.4% | 58.1–66.5% |
| 50M states | 375 | 75.0% | 71.0–78.6% |

Against the stock heuristic, the seed won 39.8% and the 10M checkpoint 56.0%.
That is close to the corrected-heuristic numbers, so the corrected heuristic is
used as the single yardstick. Result files are `logs/*eval*.json`.

## Pipeline

| Step | What | Output |
| --- | --- | --- |
| 1 | Imitation: 50,000 heuristic battles, 15 passes | `checkpoints/seed/imitation50k-best.pt` (94.97% held-out agreement) |
| 2 | Add move-category inputs (outputs unchanged) | `checkpoints/seed/imitation50k-best-category.pt` |
| 3 | Critic-only warmup, 11 updates | `checkpoints/warmup-category/learner_update_000000011.pt` |
| 4 | PPO self-play to 250M states | `checkpoints/ppo-readme/` |
| 5 | Evaluate milestones, by hand | `logs/ppo-readme-eval-*.json` |

The commands below are the ones this run used. They use this run's directory
names, so rerunning them here continues this run. For a separate reproduction,
pick new `--checkpoint-dir` names.

## Setup

This run used an RTX 3070 (8 GB), a 32-thread CPU, Windows 11 and Python 3.12.
The virtual environment is in the parent project, and Showdown is a separate
checkout at revision `7e743761cd13cdb361fc0f58eb5a866284894f5b` with the
`rlspawn` plugin built in.

```powershell
Set-Location C:/Pokemon/ps-ppo/ps-ppo-15token
$python = '../.venv/Scripts/python.exe'
$server = 'C:/Pokemon/pokemon-showdown'
```

### New Python environment

```powershell
python -m venv .venv
# Install a CUDA PyTorch 2.7.1 wheel first, then:
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` records this workstation's exact package versions.

### New Showdown checkout

Training needs the bundled `rlspawn.ts` plugin compiled into Showdown.
Collection and evaluation use ordinary challenges.

```powershell
git clone https://github.com/smogon/pokemon-showdown.git showdown
git -C showdown checkout 7e743761cd13cdb361fc0f58eb5a866284894f5b
Push-Location showdown; npm ci; Pop-Location
Copy-Item pokemon-showdown/server/chat-plugins/rlspawn.ts showdown/server/chat-plugins/rlspawn.ts
Push-Location showdown; node build; Pop-Location
$server = (Resolve-Path showdown).Path
```

`run_local.py` starts and stops its own Showdown processes. It refuses ports that
are already in use. With `--servers N` it starts N servers on consecutive ports
from `--port` (default 8002), one at a time. The first server builds Showdown and
the rest skip the build. Starting them together crashed servers, because
simultaneous builds rewrote files that other servers were reading.

## Step 1: Imitation

### What this run used

The imitation model was trained in the parent project (`C:/Pokemon/ps-ppo`) and
copied here. **Keep these parent files**; they are the training data and the
source of the seed:

| Parent file | Contents |
| --- | --- |
| `../logs/imitation-both-50000/dataset-manifest.json` | List of the 50 training shards |
| `../checkpoints/tuning/imitation-both-50000/collected-*-dataset.pt` | 50 shards of 1,000 battles each, both players recorded (100,000 trajectories, 38.9 GB) |
| `../checkpoints/tuning/teacher-large-20261116-dataset.pt` | Validation source. After a seed-71 shuffle, games 800–999 give 200 battles and 4,549 decisions. |
| `../checkpoints/imitation-both-50000/best.pt` | Best of 15 passes (pass 13). Identical to `checkpoints/seed/imitation50k-best.pt` (SHA-256 `cbce0068…`). |

Both sides of each battle were played by the corrected `SimpleHeuristicsPlayer`,
and both sides were recorded. The two trajectories from one battle always stay in
the same split.

Training settings:
- Starts from fresh weights.
- 15 passes over shuffled shards, with states shuffled within each shard.
- 64-state batches.
- AdamW, learning rate 1e-4, weight decay 0.01, gradient clip 0.5. The schedule
  warms up for 1,000 optimizer steps, holds until step 20,000, then decays until
  step 500,000.
- Cross-entropy loss with label smoothing 0.1.
- The critic is frozen.

The best model reached **94.97%** held-out agreement with the teacher: 97.3% on
move choices, 88.4% on switches and 88.4% on Tera. Its encoder did not read move
categories. Step 2 adds them.

### Rerun with this project's code

`imitation.py` defaults to the same settings. To train on the same data:

```powershell
& $python imitation.py --manifest ../logs/imitation-both-50000/dataset-manifest.json --validation-data ../checkpoints/tuning/teacher-large-20261116-dataset.pt --validation-start 800 --epochs 15 --output checkpoints/imitation-50k
```

To collect new data instead:

```powershell
& $python run_local.py --server-dir $server --entry collect --battles 50000 --validation-battles 200 --output data/teacher-50k
& $python imitation.py --manifest data/teacher-50k/manifest.json --epochs 15 --output checkpoints/imitation-50k
```

These commands have not been run end to end here; the seed came from the parent
run. A model trained here reads move categories from the start, so it skips
Step 2, and its results may differ slightly from the seed's.

Details:
- `--battles` counts training battles. The validation battles are additional.
- Shards hold 1,000 battles each (`--chunk-battles`).
- `best.pt`, `last.pt` and `status.json` are written to `--output`.
- Add `--resume` to continue an interrupted run.

## Step 2: Move-category inputs

The README-era encoder read each move's Physical/Special/Status one-hot from
move-scalar slots 3–5. The current observation schema stores the category
elsewhere, so the seed never saw it. `ModelConfig.move_category_inputs` (the
default for new models) copies the category back into slots 3–5 at model input.
Recorded datasets need no recollection. Checkpoints saved before this setting
existed load with it off and behave exactly as before.

`upgrade_checkpoint.py` enables the setting and zeroes the three new input
weights, so outputs are unchanged. It verifies this on real observations before
writing the new file, and it leaves the source file untouched:

```powershell
& $python upgrade_checkpoint.py checkpoints/seed/imitation50k-best.pt --output checkpoints/seed/imitation50k-best-category.pt --observations ../checkpoints/tuning/teacher-large-20261116-dataset.pt --max-decisions 2000
```

Result: the largest output difference was 0.0 across 2,000 decisions. The record
is in `checkpoints/seed/imitation50k-best-category.json`.

A fine-tune intended to teach the model to use the new inputs did not help, so
the upgraded seed was used unchanged. That fine-tune ran at learning rate 1e-5,
constant, for up to 3 passes. Over 53 checks its held-out agreement stayed
between 93.45% and 94.42%, below the starting 94.97%, and it was stopped. Its
history is in `logs/imitation50k-category-finetune-status.json`. The warmup was
launched from that run's `best.pt`, which holds weights identical to
`imitation50k-best-category.pt`.

## Step 3: Critic warmup

The imitation critic was frozen, so the value head starts untrained. Warmup
trains only the critic token and value head while the policy stays fixed:

```powershell
& $python run_local.py --server-dir $server --servers 4 --workers 4 --concurrency 128 --matmul-precision high --mode warmup --resume-from checkpoints/seed/imitation50k-best-category.pt --lr-schedule constant --max-updates 11 --checkpoint-dir checkpoints/warmup-category
```

The warmup ran 11 updates, about 360,700 states. Critic loss fell from 4.89 to
2.43 and was flat after about 8 updates. The original run used `--max-updates 30`
and was stopped with the stop file after update 11; `--max-updates 11` runs the
same length.

## Step 4: PPO self-play

```powershell
& $python run_local.py --server-dir $server --servers 6 --workers 6 --concurrency 192 --matmul-precision high --mode ppo --resume-from checkpoints/warmup-category/learner_update_000000011.pt --total-steps 250000000 --checkpoint-dir checkpoints/ppo-readme
```

**Rollouts:**
- Six Showdown servers run on ports 8002–8007, with one rollout worker per server.
- 192 battles run at once, 32 per worker.
- Both sides use the current policy.
- Rewards are +1 for a win, -1 for a loss and ±0.1 per faint. Ties score 0.

**Learner:** the README-era recipe is the default, so no learner flags are needed.

| Setting | Value |
| --- | --- |
| New states per update | 32,768 |
| Minibatch | 4,096 states, accumulated from 512-state microbatches |
| Passes per update | Up to 3 |
| KL early stop | Historical rule, target 0.02 |
| Learning rate | 1e-4, historical schedule |
| Backbone, policy and value rates | Equal |
| Clip / entropy / value coefficients | 0.2 / 0.02 / 0.5 |
| GAE | gamma 0.9999, lambda 0.75 |
| Value head | 51 bins, -1.6 to +1.6 |

`--matmul-precision high` enables TF32 matrix math. It was validated in
`logs/precision-validation.json`, and full FP32 is about 26% slower.

**Throughput:** about 1,450–1,500 new states per second, so 250M states take
about 47 hours on this workstation.

**Typical training values:**
- KL 0.03–0.05 and clip fraction 0.15–0.19.
- Entropy 0.3–0.54. It rose until about 20M states, then fell back to about 0.46.
- Critic loss about 2.4.
- The KL stop usually triggers at the first point it is allowed, so an update
  takes 9 optimizer steps: one full pass plus one minibatch.

**Keep the default worker queue.** `--worker-queue-episodes 400` lets workers
run ahead of the learner. With six workers that fed the learner stale data: KL
rose to 0.11–0.12 and 20% of samples were clipped.

**Checkpoints and milestones:**
- A full learner checkpoint is saved every 25 updates, and the last 3 are kept.
- Every 10M states (`--milestone-every`) a permanent policy-only copy, about
  233 MB, is written to `checkpoints/ppo-readme/milestones/states-<N>.pt`.

**Monitor, stop and resume:**
- Live status is in `logs/status.json`, and the learner log is the run's
  stdout.
- To stop cleanly, run `./Stop-LocalTraining.ps1` (or
  `New-Item -ItemType File logs/stop-training`). The run saves a checkpoint and
  shuts down its own servers.
- To resume, rerun the same command. The latest checkpoint in `--checkpoint-dir`
  takes precedence over `--resume-from`.
- A resume restores the optimizer, scheduler, RNG state, pending data and
  counters. It refuses a changed `--learning-rate`, `--lr-schedule` or
  `--lr-multipliers`.
- `--total-steps` counts new player decisions from both sides. Repeated passes
  do not count again.

### How this run actually went

Every stop and resume restored the learner exactly. The learning recipe never
changed; only the rollout setup did.

| Updates | States | Rollout setup |
| --- | --- | --- |
| 1–83 | 0–2.73M | Pilot: 4 servers, 4 workers, 128 battles, `--worker-queue-episodes 400` |
| 84–92 | 2.73–3.02M | 6 servers, 6 workers, 192 battles, queue 400. KL spiked, so the run was stopped. |
| 93–176 | 3.02–5.80M | Default queue. Workers slowed down as they aged (fixed below). |
| 177 onward | 5.80M onward | The command above, with the worker fix |

## Step 5: Evaluation

Evaluation is run by hand on milestones, during or after training:

```powershell
& $python run_local.py --server-dir $server --port 8010 --entry evaluate --checkpoint checkpoints/ppo-readme/milestones/states-000050000000.pt --battles 500 --device cuda --output logs/ppo-readme-eval-50m.json
```

- Each battle uses a fresh random team. Actions are sampled from the policy with
  seed 71.
- 500 battles take about 2½ minutes on the GPU, which briefly slows training.
- Use a port outside the training servers' range.
- The opponent defaults to the corrected heuristic. `--opponent stock` restores
  poke-env 0.15's original stat estimate, which predates the April 2026
  correction and the upstream README's result.
- `--greedy` always picks the policy's top action instead of sampling. Keep one
  protocol when comparing checkpoints.
- Results include wins, losses, ties and a 95% Wilson interval.

## What changed

Compared with the upstream repository and with earlier local attempts:

- **Architecture:** the 15-token model from upstream commit `de8cc69`, preserved
  in `vendor/`. Its tokens are Actor, Critic, Field and 12 Pokemon. Width 1024,
  2 layers, 8 heads, about 58M parameters. The actor and critic cannot attend to
  each other, and state tokens cannot attend to them. There is no recurrence.
- **Move categories:** restored as described in Step 2.
- **README-era PPO defaults:** see Step 4.
  - Earlier local runs used 64-state minibatches, a per-minibatch KL stop and
    0.5/1/2 rates for backbone/policy/value. That meant about 1,500 small
    optimizer steps per update, made the learner the bottleneck, and gave a 3–4
    week estimate for 250M states. The KL rule also cut most updates to 10–30%
    of their data.
  - Those settings are still available: `--batch-size 64 --kl-stop-rule
    minibatch --lr-multipliers current`.
- **Minibatch remainders:** a remainder smaller than half a minibatch joins the
  last minibatch instead of taking its own noisy step. Every state is trained
  on. The README-era code dropped remainders.
- **Gradient accumulation:** `--microbatch-size` only limits GPU memory. Clipping,
  optimizer steps and KL checks still happen once per full minibatch. The
  benchmark is in `logs/learner-microbatch-benchmark.json`: 512 peaks at 2.9 GB,
  while 2048 runs out of memory on 8 GB.
- **Worker slowdown fix (`worker.py`, `FinishedRoomFilter`):**
  - The bug: poke-env 0.15 waits forever for messages from battle rooms it has
    already forgotten. Showdown sends `|deinit` after each battle ends, so every
    finished battle left a stuck handler, and every new battle woke all of them.
    Worker CPU per state grew as the processes aged.
  - The fix: the workers now drop messages for finished rooms.
  - The result: throughput holds steady instead of slowing down.
- **Multi-server launcher:** servers start one at a time, as described in Setup.
- **Milestones, resume guard and `--opponent stock`:** as described above.
- **Runtime choices:** FP32 learner and inference, with optional TF32 matrix
  math. Sampling temperature is 1, so the log probabilities recorded during
  rollouts match the ones used in updates. Observations use the corrected
  current schema and turn scaling.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | 15-token policy, observation adapter, move-category inputs, parameter groups |
| `ppo_core.py` | Legal-action sampling, GAE, rollout storage, imitation loss, clipped PPO, minibatch bounds |
| `config.py`, `optimizers.py` | Architecture and runtime settings, freezing, FP32 AdamW and schedule |
| `obs_assembler.py`, `obs_*.py` | Per-player observations, move slots, transitions, action mappings |
| `poke_env_compat.py` | Correct owner Tera-type parsing |
| `utils.py`, `vocab.json`, `type_schema.json` | Encoding utilities and assets |
| `collect.py` | Both-player heuristic data in 1,000-battle shards, plus separate validation battles |
| `imitation.py`, `metrics.py` | Offline imitation, action-agreement metrics, resumable checkpoints |
| `upgrade_checkpoint.py` | Output-preserving conversion to move-category inputs |
| `inference.py` | Batched policy inference and weight synchronization |
| `worker.py` | Live battles, corrected heuristic, rewards, trajectories, finished-room filter |
| `learner.py` | PPO and critic learning, resume guard, milestones |
| `train.py` | Ray orchestration, server port assignment, status file, stop file |
| `evaluate.py` | Fixed-checkpoint evaluation against the corrected or stock heuristic |
| `checkpoint_io.py` | Strict architecture loading, atomic writes, legacy defaults |
| `run_local.py` | Starts and stops its own Showdown servers around any entry point |
| `Stop-LocalTraining.ps1` | Requests a clean stop |
| `pokemon-showdown/server/chat-plugins/rlspawn.ts` | Showdown plugin for continuous rollout battles |
| `vendor/` | Historical `de8cc69` source with tested hashes |
| `provenance.json` | Hashes of the sources copied from the parent project, and the adaptations made |
| `tests/` | 42 unit tests and fixtures |

**Checkpoints:**
- `checkpoints/seed/` holds the 50k imitation seed and its upgraded copy.
- `checkpoints/warmup-category/` holds the warmup result.
- `checkpoints/ppo-readme/` holds the PPO run.
- `checkpoints/seed/imitation-best.pt` is an older 10k-battle imitation seed:
  69.0% held-out agreement and 15.0% against the heuristic. It is still the
  default for `train.py` and `evaluate.py` when no checkpoint is given, so
  always pass `--resume-from` or `--checkpoint`.

## Tests

```powershell
& $python -m unittest discover -s tests -v
& $python run_local.py --server-dir $server --device cpu --mode ppo --model-config tests/tiny_model.json --no-resume --steps-per-update 64 --batch-size 16 --microbatch-size 8 --epochs 1 --max-updates 2 --checkpoint-dir checkpoints/smoke --save-every 1
```

The 42 tests cover:
- checkpoint parity and token isolation;
- move-category upgrades;
- README-era defaults and the resume guard;
- minibatch remainders and milestones;
- the stock heuristic;
- the finished-room filter, including a reproduction of poke-env's hang.

The second command is a two-update CPU smoke run. `tests/tiny_model.json` keeps
the 15-token layout with smaller widths.
