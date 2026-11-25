"""Collect bounded paired heuristic trajectories with separate validation games."""
import argparse
import asyncio
from pathlib import Path
import secrets
from poke_env import AccountConfiguration
from pokemon_agent.env.patches import apply_all
from pokemon_agent.env.players import RecordingTeacher
from pokemon_agent.env.showdown import make_server_conf
from pokemon_agent.files import atomic_json, sha256_file
from pokemon_agent.paths import VOCAB_PATH, data_dir
from pokemon_agent.training.datasets import new_manifest, write_shard


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="pokemon-agent-collect", description=__doc__)
    parser.add_argument("--battles", type=int, default=1000, help="Physical training battles; both sides recorded")
    parser.add_argument("--validation-battles", type=int, default=200)
    parser.add_argument("--chunk-battles", type=int, default=1000)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default=str(data_dir()/"teacher"))
    args = parser.parse_args(argv)
    if min(args.battles, args.validation_battles, args.chunk_battles, args.concurrency) < 1:
        parser.error("Battle, validation, chunk, and concurrency counts must be positive")
    return parser, args


async def collect_chunk(configuration, battles: int, concurrency: int):
    """Two recording teachers play `battles` games; returns (episodes, groups) with both sides of each game."""
    run_id = secrets.token_hex(4)
    common = dict(server_configuration=configuration, battle_format="gen9randombattle",
        max_concurrent_battles=concurrency, log_level=40)
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
    return episodes, groups


async def main(argv=None):
    parser, args = parse_args(argv)
    apply_all()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        parser.error("Output already contains a manifest; use a new collection directory")
    manifest = new_manifest(args.battles, args.validation_battles, sha256_file(VOCAB_PATH))
    configuration = make_server_conf("127.0.0.1", args.port)
    try:
        for split, count in (("validation", args.validation_battles), ("training", args.battles)):
            for start in range(0, count, args.chunk_battles):
                battles = min(args.chunk_battles, count-start)
                prefix = f"{split}-{start//args.chunk_battles:03d}"
                episodes, groups = await collect_chunk(configuration, battles, args.concurrency)
                manifest["shards"].append(write_shard(directory, prefix, split, episodes, groups))
                atomic_json(manifest_path, manifest)
                print(f"Collected {split} battles {start+battles}/{count}", flush=True)
                del episodes
        manifest["status"] = "complete"
        atomic_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        atomic_json(manifest_path, manifest)
        raise


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
