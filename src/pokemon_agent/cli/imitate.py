"""Offline, bounded-memory imitation with validation and resumable chunk saves."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import random
import torch
from pokemon_agent.config import LearnerConfig
from pokemon_agent.model.serialization import load_weights, read_checkpoint
from pokemon_agent.model.transformer import make_model
from pokemon_agent.files import atomic_json
from pokemon_agent.paths import checkpoints_dir
from pokemon_agent.rl.buffer import make_dataset
from pokemon_agent.training.checkpoints import restore_imitation_checkpoint
from pokemon_agent.training.datasets import (fingerprint, load_manifest, load_validation_episodes,
    shard_path, split_entries)
from pokemon_agent.training.imitation import (resolve_architecture, score_initial_weights,
    score_training_shards, train_imitation)
from pokemon_agent.training.optimizer import make_optimizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="pokemon-agent-imitate", description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(checkpoints_dir()/"imitation"))
    parser.add_argument("--model-config")
    parser.add_argument("--init-checkpoint",
        help="Start from these weights and architecture with a fresh optimizer (e.g. an upgraded checkpoint)")
    parser.add_argument("--learning-rate", type=float, default=LearnerConfig.lr)
    parser.add_argument("--lr-schedule", choices=("historical", "constant"), default="historical")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-data", help="Legacy single-file validation source; shuffled with seed 71")
    parser.add_argument("--validation-start", type=int, default=800)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("Epoch, batch sizes and learning rate must be positive")
    if args.init_checkpoint and args.model_config:
        parser.error("--init-checkpoint supplies the architecture; omit --model-config")
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(71)
    random.seed(71)
    manifest_path, manifest = load_manifest(args.manifest)
    if manifest.get("status", "complete") != "complete":
        parser.error("Collection must be complete before offline training")
    entries = manifest["shards"]
    training = [shard_path(manifest_path, entry) for entry in split_entries(entries, "training")]
    try:
        validation_episodes = load_validation_episodes(manifest_path, entries,
            args.validation_data, args.validation_start)
    except ValueError as exc:
        parser.error(str(exc))
    if not training or not validation_episodes:
        parser.error("Nonempty, separate training and validation data are required")
    validation = make_dataset(validation_episodes)
    heldout_hashes = {fingerprint(episode) for episode in validation_episodes}
    del validation_episodes
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    last = directory/"last.pt"
    if last.exists() and not args.resume:
        parser.error("Output already has a checkpoint; use --resume or a new directory")
    saved = read_checkpoint(last) if args.resume and last.exists() else None
    # A resumed run continues from last.pt; the initial weights are only used once.
    initial = read_checkpoint(args.init_checkpoint) if args.init_checkpoint and not saved else None
    architecture = resolve_architecture(saved, initial, args.model_config)
    model = make_model(architecture).to(args.device)
    cfg = replace(LearnerConfig(), mode="imitation", device=args.device, minibatch_size=args.batch_size,
        update_epochs=1, lr=args.learning_rate, lr_schedule=args.lr_schedule)
    optimizer, scheduler = make_optimizer(model, cfg)
    start_epoch, next_shard, best, history = 1, 0, -1., []
    if initial:
        load_weights(model, initial)
        del initial
    if saved:
        start_epoch, next_shard, best, history = restore_imitation_checkpoint(
            saved, model, optimizer, scheduler, args.device)
        del saved
    status_path = directory/"status.json"
    report = dict(status="training", manifest=str(manifest_path), epochs_requested=args.epochs,
        model_config=asdict(architecture), validation_decisions=len(validation), validation_history=history,
        init_checkpoint=str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        learning_rate=args.learning_rate, lr_schedule=args.lr_schedule, batch_size=args.batch_size)
    if history and history[0]["epoch"] == 0:
        report["initial_validation"] = history[0]["metrics"]
    common = dict(directory=directory, architecture=asdict(architecture), report=report, status_path=status_path)
    try:
        if args.init_checkpoint and not history:
            best = score_initial_weights(model, validation, history, **common)
        best = train_imitation(model, optimizer, scheduler, cfg, training, validation, heldout_hashes,
            epochs=args.epochs, start_epoch=start_epoch, next_shard=next_shard, best=best, history=history,
            **common)
        report["status"] = "evaluating_training"
        atomic_json(status_path, report)
        report["training_metrics"] = score_training_shards(model, training)
        report["status"] = "complete"
        atomic_json(status_path, report)
        print(json.dumps(dict(status="complete", training_metrics=report["training_metrics"], best_heldout_accuracy=best)))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        atomic_json(status_path, report)
        raise


if __name__ == "__main__":
    main()
