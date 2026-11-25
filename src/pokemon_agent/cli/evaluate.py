"""Play a fixed 15-token checkpoint against the heuristic (corrected or stock); no training."""
import argparse
import asyncio
import json
import random
import secrets
import time
import numpy as np
import torch
from poke_env import AccountConfiguration
from pokemon_agent.env.heuristics import OPPONENTS
from pokemon_agent.env.patches import apply_all
from pokemon_agent.env.players import SavedModelPlayer
from pokemon_agent.env.showdown import make_server_conf
from pokemon_agent.files import atomic_json
from pokemon_agent.model.serialization import load_model
from pokemon_agent.paths import default_seed_checkpoint, logs_dir
from pokemon_agent.training.metrics import wilson


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="pokemon-agent-evaluate", description=__doc__)
    parser.add_argument("--checkpoint", default=str(default_seed_checkpoint()))
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy", action="store_true", help="Default is sampled actions, matching the recorded baseline")
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="corrected",
        help="corrected: Gen 9 level-aware stat estimate used for collection; stock: poke-env's original")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", default=str(logs_dir()/"evaluation.json"))
    args = parser.parse_args(argv)
    if min(args.battles, args.concurrency) < 1:
        parser.error("Battles and concurrency must be positive")
    return args


async def main(argv=None):
    args = parse_args(argv)
    apply_all()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = load_model(args.checkpoint, args.device)
    model.requires_grad_(False)
    configuration = make_server_conf("127.0.0.1", args.port)
    common = dict(server_configuration=configuration, battle_format="gen9randombattle",
        max_concurrent_battles=args.concurrency, log_level=40)
    report = dict(status="running", checkpoint=args.checkpoint, target_battles=args.battles,
        action_selection="greedy" if args.greedy else "sampled", opponent=f"{args.opponent} SimpleHeuristicsPlayer",
        policy_seed=args.seed, teams="fresh simulator-generated random teams", training=False)
    started = time.monotonic()

    def progress(player):
        report.update(battles=player.n_finished_battles, wins=player.n_won_battles,
            losses=player.n_lost_battles, ties=player.n_tied_battles, model_decisions=player.decisions,
            win_rate=player.n_won_battles/player.n_finished_battles,
            win_rate_95pct_wilson=wilson(player.n_won_battles, player.n_finished_battles),
            seconds=time.monotonic()-started)
        try:
            atomic_json(args.output, report)
        except OSError:
            # Progress persistence must not interrupt a battle's completion.
            pass

    run_id = secrets.token_hex(4)
    bot = SavedModelPlayer(model, args.device, greedy=args.greedy, progress=progress,
        account_configuration=AccountConfiguration(f"eval{run_id}a", None), **common)
    opponent = OPPONENTS[args.opponent](
        account_configuration=AccountConfiguration(f"eval{run_id}b", None), **common)
    try:
        await asyncio.wait_for(bot.battle_against(opponent, n_battles=args.battles), timeout=max(600, args.battles*30))
        if bot.n_finished_battles != args.battles:
            raise RuntimeError("Evaluation completed with an incomplete battle count")
        report["status"] = "complete"
        progress(bot)
        atomic_json(args.output, report)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        atomic_json(args.output, report)
        raise
    finally:
        await asyncio.gather(bot.ps_client.stop_listening(), opponent.ps_client.stop_listening())


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
