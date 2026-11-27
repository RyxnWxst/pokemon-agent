# ps-ppo: 15-token reproduction

This project reproduces the upstream ps-ppo README result: a 15-token Transformer
policy is trained by imitating `SimpleHeuristicsPlayer`, then trained by PPO
self-play for about 250 million states. The upstream README reports more than
85% wins against that heuristic.

All Python code lives in this directory. The Python environment and a built
Pokemon Showdown server are external dependencies. So is the imitation dataset,
which lives in the parent project (see [Step 1](#step-1-imitation)).

## Results

The self-play run reached the upstream README's target. It was stopped at
200.8M states. Each row is 500 battles against the corrected heuristic, with
actions sampled from the policy, unless noted.

| Checkpoint | Wins | Win rate | 95% interval |
| --- | --- | --- | --- |
| Imitation seed | 210 | 42.0% | 37.8–46.4% |
| PPO update 75 (2.46M states) | 213 | 42.6% | 38.3–47.0% |
| 10M states | 286 | 57.2% | 52.8–61.5% |
| 20M states | 312 | 62.4% | 58.1–66.5% |
| 50M states | 375 | 75.0% | 71.0–78.6% |
| 80M states | 403 | 80.6% | 76.9–83.8% |
| 100M states (1,000 battles) | 799 | 79.9% | 77.3–82.3% |
| 110M states | 407 | 81.4% | 77.8–84.6% |
| **130M states** | 436 | **87.2%** | 84.0–89.8% |
| 180M states | 426 | 85.2% | 81.8–88.0% |
| 200M states | 431 | 86.2% | 82.9–88.9% |

The 130M checkpoint scored highest and is saved as
`checkpoints/best/ppo-readme-130m.pt`. Its range overlaps those of 180M and
200M, so the three are statistically indistinguishable at 500 battles.

Against the stock heuristic, the seed won 39.8% and the 10M checkpoint 56.0%.
That is close to the corrected-heuristic numbers, so the corrected heuristic is
used as the single yardstick. Result files are `logs/*eval*.json`.

Follow-up experiments from this model (reward fixes, training against the
heuristic, learning-rate decay) are described in
[Experiments after 200M](#experiments-after-200m).

## Pipeline

| Step | What | Output |
| --- | --- | --- |
| 1 | Imitation: 50,000 heuristic battles, 15 passes | `checkpoints/seed/imitation50k-best.pt` (94.97% held-out agreement) |
| 2 | Add move-category inputs (outputs unchanged) | `checkpoints/seed/imitation50k-best-category.pt` |
| 3 | Critic-only warmup, 11 updates | `checkpoints/warmup-category/learner_update_000000011.pt` |
| 4 | PPO self-play (target 250M states, stopped at 200.8M) | `checkpoints/ppo-readme/` |
| 5 | Evaluate milestones, by hand | `logs/ppo-readme-eval-*.json` |

The commands below are the ones this run used. They use this run's directory
names, so rerunning them here continues this run. For a separate reproduction,
pick new `--checkpoint-dir` names.

This run used the faint reward as it was before the
[faint-reward fix](#1-faint-reward-fix). The fixed reward is now the default,
so the warmup and PPO commands below pass `--faint-reward legacy` to match the
run exactly.

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
& $python run_local.py --server-dir $server --servers 4 --workers 4 --concurrency 128 --matmul-precision high --mode warmup --resume-from checkpoints/seed/imitation50k-best-category.pt --lr-schedule constant --max-updates 11 --faint-reward legacy --checkpoint-dir checkpoints/warmup-category
```

The warmup ran 11 updates, about 360,700 states. Critic loss fell from 4.89 to
2.43 and was flat after about 8 updates. The original run used `--max-updates 30`
and was stopped with the stop file after update 11; `--max-updates 11` runs the
same length.

## Step 4: PPO self-play

```powershell
& $python run_local.py --server-dir $server --servers 6 --workers 6 --concurrency 192 --matmul-precision high --mode ppo --resume-from checkpoints/warmup-category/learner_update_000000011.pt --total-steps 250000000 --faint-reward legacy --checkpoint-dir checkpoints/ppo-readme
```

**Rollouts:**
- Six Showdown servers run on ports 8002–8007, with one rollout worker per server.
- 192 battles run at once, 32 per worker.
- Both sides use the current policy.
- Rewards are +1 for a win, -1 for a loss and ±0.1 per faint, with ties
  scoring 0. As run, the faint part was broken; see the
  [faint-reward fix](#1-faint-reward-fix).

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
| 177–6,123 | 5.80–200.8M | The command above, with the worker fix. Stopped at 200.8M. |

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

## Experiments after 200M

These experiments started on Nov 27, 2026, from the better of the 130M and
200M checkpoints. They test four ideas, one stage after another, using
`experiments.py`, which runs unattended.

### 1. Faint-reward fix

The faint reward is meant to give -0.1 when one of our Pokemon faints and
+0.1 when an opponent's does.
- **The bug:** a later upstream change to the observation layout made it read
  the wrong feature: index 125, "ability not yet known", instead of 113,
  "fainted". Upstream's README-era code (`de8cc69`) read the right one.
- **What it paid,** measured on 1,000 recorded games: never a penalty for our
  own faints, and +0.1 about 3 times per game, whenever a new opponent Pokemon
  with an unknown ability appeared. That averaged +0.31 per game for wins and
  losses alike, so the 200M run learned almost entirely from win/loss reward.
- **Now:** the fixed reward is the default and pays +0.09 per game on average
  for wins and -0.10 for losses. `--faint-reward legacy` reproduces the old
  behavior.

### 2. HP-lead shaping (`--hp-shaping c`)

This is potential-based shaping (Ng et al., 1999), with
Phi = c × (our mean HP fraction − the opponent's mean HP fraction).
- Each decision's reward gets gamma × Phi(next) − Phi(this) added, and Phi is
  0 once the game ends.
- Opponent Pokemon not yet seen count as full HP. Observations show them with
  0 HP, which would otherwise make every reveal look like losing the lead.
- Both teams start at full HP, so over a game the shaping adds up to almost
  exactly zero. It can't change which policy is best; it only gives the learner
  credit earlier. That matters here because GAE uses lambda = 0.75, so each
  step's credit mostly looks only a few turns ahead.
- The experiments use c = 0.5.

### 3. Training against the heuristic (`--mode ppo_vs_teacher`)

Same PPO and reward, but the opponent is the corrected `SimpleHeuristicsPlayer`
instead of the current policy, and only our side's decisions are trained on.
- It optimizes the evaluated matchup directly. That makes it the most direct
  way to raise this score, but it is no longer README-style self-play, and the
  policy may learn the heuristic's habits rather than better play overall.
- `--opponent heuristic` uses the heuristic for a critic warmup too.

### 4. Learning-rate decay (`--lr-schedule linear`)

The historical schedule counts optimizer steps and decays toward step 500,000.
KL early stopping keeps each update to about 9 steps, so after 200M states the
rate had barely moved from 1e-4.

The linear schedule warms up for `--lr-warmup-steps` optimizer steps, then
decays with new states to `--lr-final-fraction` × the rate at `--total-steps`.
It follows states rather than steps, so it resumes exactly.

### The pipeline

| Stage | Opponent | New states | Reward | Learning rate |
| --- | --- | --- | --- | --- |
| `reward-selfplay` | Self-play | 40M | Fixed faints + HP shaping (c = 0.5) | 1e-4 after warmup |
| `vs-heuristic` | Heuristic | 30M | Same | Same |
| `lr-decay` | Whichever stage produced the best model | 30M | Same | Linear decay to 0 |

- **Starting point:** each stage starts from the best model at the time it
  begins.
- **Warmup:** each stage first runs a 10-update critic-only warmup with its own
  reward and opponent, because either change shifts what the critic should
  predict.
- **Baseline:** the 130M and 200M checkpoints each get 2,000 battles, and the
  better one is the model to beat.
- **Milestone tests:** every 5M states, a milestone gets 1,000 battles against
  the corrected heuristic.
- **Promotion:** a milestone that beats the current best gets 1,000 more battles
  with a different sampling seed. It becomes the best only if the pooled
  2,000-battle result is higher. That keeps a lucky 1,000-battle result from
  being crowned.
- **Best model:** copied to `checkpoints/best-experiments/best.pt`, with its
  pooled results and the history of improvements in `best.json`.

Run, watch and stop it:

```powershell
& $python -u experiments.py                      # start, or resume after a stop or crash
Get-Content logs/experiments/status.json         # stage, phase, training progress, best so far
Get-Content checkpoints/best-experiments/best.json
New-Item -ItemType File logs/experiments/stop    # stop cleanly; training saves a checkpoint first
```

How it behaves:
- **The plan can be edited mid-run.** It is in `logs/experiments/plan.json` and
  is re-read before each stage, so later stages can be changed while it runs.
- **Results are kept as it goes.** Every evaluation is recorded in
  `logs/experiments/evaluations.json`, and logs are in `logs/experiments/`.
- **A restart resumes.** Finished stages and evaluations are not repeated, and
  a crashed training run is retried up to three times from its own checkpoints.
- **Training files:** each stage's checkpoints are in
  `checkpoints/experiments/<stage>/`.

Self-play runs at about 1,500 states/s, so the first stage takes about
7.5 hours. Heuristic games yield half the training states per battle, so the
later stages will be slower. The whole plan should take about 1.5 days.

Results will be added here as the stages finish.

## What changed

Compared with the upstream repository and with earlier local attempts:

- **Architecture:** the 15-token model from upstream commit `de8cc69`, preserved
  in `vendor/`. Its tokens are Actor, Critic, Field and 12 Pokemon. Width 1024,
  2 layers, 8 heads, about 58M parameters. The actor and critic cannot attend to
  each other, and state tokens cannot attend to them. There is no recurrence.
- **Move categories:** restored as described in Step 2.
- **Faint reward, HP shaping, heuristic opponent and linear learning-rate
  decay:** see [Experiments after 200M](#experiments-after-200m).
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
| `worker.py` | Live battles, corrected heuristic, trajectories, finished-room filter |
| `rewards.py` | Terminal and faint rewards, potential-based HP-lead shaping |
| `learner.py` | PPO and critic learning, resume guard, milestones, linear schedule progress |
| `train.py` | Ray orchestration, server port assignment, reward/opponent/schedule options, status and stop files |
| `experiments.py` | Unattended staged experiments with automatic evaluation and best-model tracking |
| `evaluate.py` | Fixed-checkpoint evaluation against the corrected or stock heuristic |
| `checkpoint_io.py` | Strict architecture loading, atomic writes, legacy defaults |
| `run_local.py` | Starts and stops its own Showdown servers around any entry point (`--skip-build` reuses the built server) |
| `Stop-LocalTraining.ps1` | Requests a clean stop |
| `pokemon-showdown/server/chat-plugins/rlspawn.ts` | Showdown plugin for continuous rollout battles |
| `vendor/` | Historical `de8cc69` source with tested hashes |
| `provenance.json` | Hashes of the sources copied from the parent project, and the adaptations made |
| `tests/` | 55 unit tests and fixtures |

**Checkpoints:**
- `checkpoints/seed/` holds the 50k imitation seed and its upgraded copy.
- `checkpoints/warmup-category/` holds the warmup result.
- `checkpoints/ppo-readme/` holds the PPO run.
- `checkpoints/best/ppo-readme-130m.pt` is the self-play run's best 500-battle
  result.
- `checkpoints/experiments/` holds the experiment stages, and
  `checkpoints/best-experiments/` holds their best model.
- `checkpoints/seed/imitation-best.pt` is an older 10k-battle imitation seed:
  69.0% held-out agreement and 15.0% against the heuristic. It is still the
  default for `train.py` and `evaluate.py` when no checkpoint is given, so
  always pass `--resume-from` or `--checkpoint`.

## Tests

```powershell
& $python -m unittest discover -s tests -v
& $python run_local.py --server-dir $server --device cpu --mode ppo --model-config tests/tiny_model.json --no-resume --steps-per-update 64 --batch-size 16 --microbatch-size 8 --epochs 1 --max-updates 2 --checkpoint-dir checkpoints/smoke --save-every 1
```

The 55 tests cover:
- checkpoint parity and token isolation;
- move-category upgrades;
- README-era defaults and the resume guard;
- minibatch remainders and milestones;
- the stock heuristic;
- the finished-room filter, including a reproduction of poke-env's hang;
- the faint-index fix and legacy reward;
- HP shaping;
- the heuristic opponent and the linear schedule;
- the experiment plan and best-model promotion.

The second command is a two-update CPU smoke run. `tests/tiny_model.json` keeps
the 15-token layout with smaller widths.
