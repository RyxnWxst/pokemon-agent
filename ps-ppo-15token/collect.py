"""Collect bounded paired heuristic trajectories with separate validation games."""
import argparse
import asyncio
import hashlib
from pathlib import Path
import secrets
import numpy as np
import torch
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from checkpoint_io import atomic_json, replace_file
from evaluate import SavedModelPlayer
from config import ROOT


class RecordingTeacher(SavedModelPlayer, SimpleHeuristicsPlayer):
    def __init__(self, **kwargs):
        self.pending, self.episodes = {}, {}
        super().__init__(None, "cpu", **kwargs)

    def choose_move(self, battle):
        order = SimpleHeuristicsPlayer.choose_move(self, battle)
        observation = self.assembler.assemble(battle, events=self.events.pop(battle.battle_tag, []))
        action = self.assembler.map_order_to_index(order, battle)
        start, end = self.assembler.offsets["action_mask"]
        if observation[start:end][action] < .5:
            raise RuntimeError("Teacher action is outside the legal mask")
        choice, kwargs = self.assembler.map_index_to_order(action, battle)
        if choice == "DEFAULT" or self.create_order(choice, **kwargs).message != order.message:
            raise RuntimeError("Teacher action failed its order roundtrip")
        self.pending.setdefault(battle.battle_tag, []).append((observation, action))
        return order

    def _battle_finished_callback(self, battle):
        samples = self.pending.pop(battle.battle_tag)
        observations = torch.from_numpy(np.stack([item[0] for item in samples]))
        actions = torch.tensor([item[1] for item in samples], dtype=torch.long)
        self.episodes[battle.battle_tag] = observations, actions, torch.zeros(len(actions))
        super()._battle_finished_callback(battle)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=1000, help="Physical training battles; both sides recorded")
    parser.add_argument("--validation-battles", type=int, default=200)
    parser.add_argument("--chunk-battles", type=int, default=1000)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default=str(ROOT/"data/teacher"))
    args = parser.parse_args()
    if min(args.battles, args.validation_battles, args.chunk_battles, args.concurrency) < 1:
        parser.error("Battle, validation, chunk, and concurrency counts must be positive")
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        parser.error("Output already contains a manifest; use a new collection directory")
    manifest = dict(status="collecting", both_sides=True, training_battles=args.battles,
        validation_battles=args.validation_battles, shards=[],
        vocab_sha256=hashlib.sha256((ROOT/"vocab.json").read_bytes()).hexdigest())
    configuration = ServerConfiguration(f"ws://127.0.0.1:{args.port}/showdown/websocket",
        f"http://127.0.0.1:{args.port}/action.php?")
    try:
        for split, count in (("validation", args.validation_battles), ("training", args.battles)):
            for start in range(0, count, args.chunk_battles):
                battles = min(args.chunk_battles, count-start)
                prefix = f"{split}-{start//args.chunk_battles:03d}"
                run_id = secrets.token_hex(4)
                common = dict(server_configuration=configuration, battle_format="gen9randombattle",
                    max_concurrent_battles=args.concurrency, log_level=40)
                left = RecordingTeacher(account_configuration=AccountConfiguration(f"data{run_id}a", None), **common)
                right = RecordingTeacher(account_configuration=AccountConfiguration(f"data{run_id}b", None), **common)
                try:
                    await asyncio.wait_for(left.battle_against(right, n_battles=battles), timeout=max(600, battles*30))
                finally:
                    await asyncio.gather(left.ps_client.stop_listening(), right.ps_client.stop_listening())
                if left.episodes.keys() != right.episodes.keys() or len(left.episodes) != battles:
                    raise RuntimeError("Paired battle recordings are incomplete")
                episodes, groups = [], []
                for tag in left.episodes:
                    groups.append(dict(battle_id=tag, episode_indices=[len(episodes), len(episodes)+1]))
                    episodes.extend((left.episodes[tag], right.episodes[tag]))
                path = directory/f"{prefix}.pt"
                temporary = path.with_suffix(".tmp")
                torch.save(episodes, temporary)
                replace_file(temporary, path)
                atomic_json(directory/f"{prefix}-groups.json", groups)
                manifest["shards"].append(dict(path=path.name, groups=f"{prefix}-groups.json", split=split,
                    battles=battles, perspectives=len(episodes), decisions=sum(len(item[1]) for item in episodes)))
                atomic_json(manifest_path, manifest)
                print(f"Collected {split} battles {start+battles}/{count}", flush=True)
                del left, right, episodes
        manifest["status"] = "complete"
        atomic_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    asyncio.run(main())
