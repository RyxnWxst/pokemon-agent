"""Play a fixed 15-token checkpoint against the heuristic (corrected or stock); no training."""
import argparse
import asyncio
import json
import math
import random
import secrets
import time
import numpy as np
import torch
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player, SimpleHeuristicsPlayer, DefaultBattleOrder
from checkpoint_io import atomic_json, load_weights, model_config, read_checkpoint
from config import ROOT
from model import make_model
from obs_assembler import ObservationAssembler
from ppo_core import masked_sample
from worker import _corrected_stat_estimation

SimpleHeuristicsPlayer._stat_estimation = _corrected_stat_estimation


def stock_stat_estimation(mon, stat):
    """poke-env 0.15 SimpleHeuristicsPlayer._stat_estimation, verbatim (before the April 2026 correction)."""
    # Stats boosts value
    if mon.boosts[stat] > 1:
        boost = (2 + mon.boosts[stat]) / 2
    else:
        boost = 2 / (2 - mon.boosts[stat])
    return ((2 * mon.base_stats[stat] + 31) + 5) * boost


class StockHeuristicsPlayer(SimpleHeuristicsPlayer):
    """The heuristic with poke-env's original stat estimate, as in the README-era evaluation.

    The corrected estimate is patched onto the shared class and the static move
    chooser looks it up there, so it is swapped only for this player's synchronous
    decision. Tera parsing (poke_env_compat) remains process-wide for both sides.
    """
    def choose_move(self, battle):
        corrected = SimpleHeuristicsPlayer.__dict__["_stat_estimation"]
        SimpleHeuristicsPlayer._stat_estimation = staticmethod(stock_stat_estimation)
        try:
            return super().choose_move(battle)
        finally:
            SimpleHeuristicsPlayer._stat_estimation = corrected


OPPONENTS = {"corrected": SimpleHeuristicsPlayer, "stock": StockHeuristicsPlayer}


class SavedModelPlayer(Player):
    def __init__(self, model, device, greedy=False, progress=None, **kwargs):
        self.model, self.device, self.greedy = model, device, greedy
        self.progress = progress
        self.assembler = ObservationAssembler()
        self.events = {}
        self.decisions = 0
        super().__init__(**kwargs)

    async def _handle_battle_message(self, split_messages):
        if split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
            tag = split_messages[0][0][1:].strip()
            self.events.setdefault(tag, []).extend(p for p in split_messages
                if len(p)>1 and p[0]=="" and p[1] not in ("t:", "request", "turn"))
        await super()._handle_battle_message(split_messages)

    async def choose_move(self, battle):
        observation = self.assembler.assemble(battle, events=self.events.pop(battle.battle_tag, []))
        obs = torch.from_numpy(observation[None]).to(self.device)
        with torch.no_grad():
            logits = self.model(obs)[0]
            start, end = self.assembler.offsets["action_mask"]
            action, *_ = masked_sample(logits, obs[:, start:end], greedy=self.greedy)
        self.decisions += 1
        choice, kwargs = self.assembler.map_index_to_order(int(action), battle)
        return DefaultBattleOrder() if choice == "DEFAULT" else self.create_order(choice, **kwargs)

    def _battle_finished_callback(self, battle):
        self.events.pop(battle.battle_tag, None)
        super()._battle_finished_callback(battle)
        if self.progress:
            self.progress(self)


def wilson(wins, count):
    z = 1.959963984540054
    p = wins/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [center-radius, center+radius]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT/"checkpoints/seed/imitation-best.pt"))
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy", action="store_true", help="Default is sampled actions, matching the recorded baseline")
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="corrected",
        help="corrected: Gen 9 level-aware stat estimate used for collection; stock: poke-env's original")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", default=str(ROOT/"logs/evaluation.json"))
    args = parser.parse_args()
    if min(args.battles, args.concurrency) < 1:
        parser.error("Battles and concurrency must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = read_checkpoint(args.checkpoint)
    model = make_model(model_config(checkpoint)).to(args.device).eval()
    load_weights(model, checkpoint)
    model.requires_grad_(False)
    del checkpoint
    configuration = ServerConfiguration(f"ws://127.0.0.1:{args.port}/showdown/websocket",
        f"http://127.0.0.1:{args.port}/action.php?")
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


if __name__ == "__main__":
    asyncio.run(main())
