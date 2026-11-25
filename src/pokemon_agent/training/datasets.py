"""Teacher-data shards on disk: the collection manifest, shard files and the held-out split."""
import hashlib
import json
from pathlib import Path
import random

import torch

from pokemon_agent.files import atomic_json, atomic_torch_save


def new_manifest(training_battles: int, validation_battles: int, vocab_sha256: str) -> dict:
    return dict(status="collecting", both_sides=True, training_battles=training_battles,
        validation_battles=validation_battles, shards=[], vocab_sha256=vocab_sha256)


def write_shard(directory, prefix: str, split: str, episodes, groups) -> dict:
    """Save one chunk of episodes and its battle grouping; returns the manifest entry."""
    directory = Path(directory)
    path = atomic_torch_save(episodes, directory/f"{prefix}.pt")
    atomic_json(directory/f"{prefix}-groups.json", groups)
    return dict(path=path.name, groups=f"{prefix}-groups.json", split=split, battles=len(groups),
        perspectives=len(episodes), decisions=sum(len(item[1]) for item in episodes))


def load_manifest(path):
    """Returns (resolved manifest path, manifest dict)."""
    path = Path(path).resolve()
    return path, json.loads(path.read_text(encoding="utf-8-sig"))


def shard_path(manifest_path, entry) -> Path:
    value = Path(entry["path"])
    return value if value.is_absolute() else Path(manifest_path).parent/value


def split_entries(entries, split: str) -> list:
    """Shards of one split; shards without a split label are training data."""
    return [entry for entry in entries if entry.get("split", "training") == split]


def load_validation_episodes(manifest_path, entries, legacy_path=None, legacy_start=800) -> list:
    """The manifest's validation shards, or the tail of a legacy single file shuffled with seed 71."""
    episodes = []
    for entry in split_entries(entries, "validation"):
        episodes.extend(torch.load(shard_path(manifest_path, entry), map_location="cpu", weights_only=False))
    if legacy_path:
        if episodes:
            raise ValueError("Choose manifest validation or --validation-data, not both")
        base = torch.load(legacy_path, map_location="cpu", weights_only=False)
        random.Random(71).shuffle(base)
        episodes = base[legacy_start:]
    return episodes


def fingerprint(episode) -> bytes:
    """Hash of an episode's observations and actions, to detect train/validation overlap."""
    digest = hashlib.sha256()
    for tensor in episode[:2]:
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.digest()


def assert_disjoint(episodes, heldout_hashes) -> None:
    if any(fingerprint(episode) in heldout_hashes for episode in episodes):
        raise RuntimeError("Training/validation episode overlap")
