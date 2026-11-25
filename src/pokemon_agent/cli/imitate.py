"""Offline, bounded-memory imitation with validation and resumable chunk saves."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import torch
from checkpoint_io import atomic_json, load_weights, model_config, read_checkpoint, replace_file
from config import ROOT, LearnerConfig, ModelConfig
from metrics import action_metrics, combine_metrics, make_dataset
from model import make_model
from optimizers import make_optimizer
from ppo_core import ppo_update


def fingerprint(episode):
    digest = hashlib.sha256()
    for tensor in episode[:2]:
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(ROOT/"checkpoints/imitation"))
    parser.add_argument("--model-config")
    parser.add_argument("--init-checkpoint",
        help="Start from these weights and architecture with a fresh optimizer (e.g. an upgraded checkpoint)")
    parser.add_argument("--learning-rate", type=float, default=LearnerConfig.lr)
    parser.add_argument("--lr-schedule", choices=("historical", "constant"), default="historical")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-data", help="Legacy single-file validation source; shuffled with seed 71")
    parser.add_argument("--validation-start", type=int, default=800)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("Epoch, batch sizes and learning rate must be positive")
    if args.init_checkpoint and args.model_config:
        parser.error("--init-checkpoint supplies the architecture; omit --model-config")
    torch.set_num_threads(2)
    torch.manual_seed(71)
    random.seed(71)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("status", "complete") != "complete":
        parser.error("Collection must be complete before offline training")
    entries = manifest["shards"]

    def path(entry):
        value = Path(entry["path"])
        return value if value.is_absolute() else manifest_path.parent/value

    training = [entry for entry in entries if entry.get("split", "training") == "training"]
    validation_episodes = []
    for entry in entries:
        if entry.get("split") == "validation":
            validation_episodes.extend(torch.load(path(entry), map_location="cpu", weights_only=False))
    if args.validation_data:
        if validation_episodes:
            parser.error("Choose manifest validation or --validation-data, not both")
        base = torch.load(args.validation_data, map_location="cpu", weights_only=False)
        random.Random(71).shuffle(base)
        validation_episodes = base[args.validation_start:]
        del base
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
    architecture = (model_config(saved) if saved else model_config(initial) if initial else
        ModelConfig(**json.loads(Path(args.model_config).read_text())) if args.model_config else ModelConfig())
    model = make_model(architecture).to(args.device)
    cfg = replace(LearnerConfig(), mode="imitation", device=args.device, minibatch_size=args.batch_size,
        update_epochs=1, lr=args.learning_rate, lr_schedule=args.lr_schedule)
    optimizer, scheduler = make_optimizer(model, cfg)
    start_epoch, next_shard, best, history = 1, 0, -1., []
    if initial:
        load_weights(model, initial)
        del initial
    if saved:
        load_weights(model, saved)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if args.device.startswith("cuda") and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch, next_shard = saved["epoch"], saved["next_shard"]
        best, history = saved["best_accuracy"], saved["validation_history"]
        del saved
    status_path = directory/"status.json"
    report = dict(status="training", manifest=str(manifest_path), epochs_requested=args.epochs,
        model_config=asdict(architecture), validation_decisions=len(validation), validation_history=history,
        init_checkpoint=str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        learning_rate=args.learning_rate, lr_schedule=args.lr_schedule, batch_size=args.batch_size)
    if history and history[0]["epoch"] == 0:
        report["initial_validation"] = history[0]["metrics"]

    def save_checkpoint(destination, epoch, next_position, resume=False):
        payload = dict(project="ps-ppo-15token-imitation", architecture="readme_15token", phase="imitation",
            model=model.state_dict(), model_config=asdict(architecture), epoch=epoch, next_shard=next_position,
            best_accuracy=best, validation_history=history)
        if resume:
            payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None)
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        replace_file(temporary, destination)

    try:
        if args.init_checkpoint and not history:
            # Score the starting weights so best.pt is never worse than the initialization.
            metrics = action_metrics(model, validation)
            history.append(dict(epoch=0, shard=0, metrics=metrics, loss=None))
            best = metrics["accuracy"]
            save_checkpoint(directory/"best.pt", 0, 0)
            report.update(initial_validation=metrics, best_validation_accuracy=best)
            atomic_json(status_path, report)
            print(json.dumps(dict(epoch=0, heldout_accuracy=best)), flush=True)
        for epoch in range(start_epoch, args.epochs+1):
            order = list(range(len(training)))
            random.Random(7100+epoch).shuffle(order)
            for position in range(next_shard if epoch == start_epoch else 0, len(order)):
                episodes = torch.load(path(training[order[position]]), map_location="cpu", weights_only=False)
                if any(fingerprint(episode) in heldout_hashes for episode in episodes):
                    raise RuntimeError("Training/validation episode overlap")
                dataset = make_dataset(episodes)
                del episodes
                report.update(epoch=epoch, shard=position+1, total_shards=len(training))
                atomic_json(status_path, report)
                stats = ppo_update(model, optimizer, dataset, scheduler=scheduler, mode="imitation",
                    **cfg.ppo_kwargs(), v_min=cfg.v_min, v_max=cfg.v_max, v_bins=cfg.v_bins)
                del dataset
                metrics = action_metrics(model, validation)
                history.append(dict(epoch=epoch, shard=position+1, metrics=metrics, loss=stats.pg_loss))
                if metrics["accuracy"] > best:
                    best = metrics["accuracy"]
                    save_checkpoint(directory/"best.pt", epoch, position+1)
                save_checkpoint(last, epoch, position+1, resume=True)
                report.update(best_validation_accuracy=best, last_validation=metrics)
                atomic_json(status_path, report)
                print(json.dumps(dict(epoch=epoch, shard=position+1, heldout_accuracy=metrics["accuracy"])), flush=True)
        report["status"] = "evaluating_training"
        atomic_json(status_path, report)
        scores = []
        for entry in training:
            episodes = torch.load(path(entry), map_location="cpu", weights_only=False)
            dataset = make_dataset(episodes)
            del episodes
            scores.append(action_metrics(model, dataset))
            del dataset
        report["training_metrics"] = combine_metrics(scores)
        report["status"] = "complete"
        atomic_json(status_path, report)
        print(json.dumps(dict(status="complete", training_metrics=report["training_metrics"], best_heldout_accuracy=best)))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        atomic_json(status_path, report)
        raise


if __name__ == "__main__":
    main()
