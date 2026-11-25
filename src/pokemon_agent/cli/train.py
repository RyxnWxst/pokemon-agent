"""Run local distributed 15-token imitation, critic warmup, or PPO self-play."""
import argparse
import asyncio
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import ray
import torch
from pokemon_agent.config import LearnerConfig, ModelConfig, RunConfig
from pokemon_agent.distributed.launch import init_ray, monitor, start_actors
from pokemon_agent.files import atomic_json
from pokemon_agent.model.serialization import model_config, read_checkpoint
from pokemon_agent.paths import checkpoints_dir, default_seed_checkpoint, logs_dir


def assign_ports(workers, ports):
    """Spread rollout workers across Showdown servers, round-robin."""
    if workers < len(ports):
        raise ValueError("Use at least one rollout worker per Showdown server")
    return [ports[index % len(ports)] for index in range(workers)]


def parse_args(argv=None):
    defaults = LearnerConfig()
    parser = argparse.ArgumentParser(prog="pokemon-agent-train", description=__doc__)
    parser.add_argument("--mode", choices=("imitation", "warmup", "ppo", "ppo_vs_teacher"), default="ppo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--ports", type=int, nargs="+",
        help="Several Showdown servers; overrides --port and assigns workers round-robin")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1, help="Independent rollout processes sharing inference")
    parser.add_argument("--worker-queue-episodes", type=int, default=0,
        help="Finished episodes each worker may hold while the learner is busy; 0 keeps the original bound")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    # Learner defaults are the README-era (de8cc69) recipe; see config.LearnerConfig.
    parser.add_argument("--steps-per-update", type=int, default=defaults.steps_per_update)
    parser.add_argument("--batch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--microbatch-size", type=int, default=defaults.microbatch_size,
        help="Accumulate this many decisions at a time within each training minibatch; 0 uses the full minibatch")
    parser.add_argument("--kl-stop-rule", choices=("minibatch", "historical"), default=defaults.kl_stop_rule,
        help="Original signed-KL epoch average after the first pass (historical), or a per-minibatch check")
    parser.add_argument("--lr-multipliers", choices=("historical", "current"), default=defaults.ppo_lr_multipliers,
        help="PPO backbone/policy/value rates: historical 1/1/1, current 0.5/1/2")
    parser.add_argument("--epochs", type=int, default=defaults.update_epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.lr)
    parser.add_argument("--lr-schedule", choices=("historical", "constant"), default=defaults.lr_schedule)
    parser.add_argument("--total-steps", type=int, default=0, help="New decisions in this phase; 0 is unlimited")
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--model-config", help="Optional JSON ModelConfig, primarily for small smoke tests")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--milestone-every", type=int, default=defaults.milestone_every_states,
        help="Keep a permanent policy-only copy every this many new states; 0 disables")
    parser.add_argument("--status-file", default=str(logs_dir() / "status.json"))
    parser.add_argument("--stop-file", default=str(logs_dir() / "stop-training"))
    args = parser.parse_args(argv)
    if min(args.concurrency, args.workers, args.steps_per_update, args.batch_size, args.epochs, args.save_every) < 1:
        parser.error("Concurrency and batch/update/save sizes must be positive")
    if args.workers > args.concurrency:
        parser.error("Worker count cannot exceed concurrent battles")
    if args.microbatch_size < 0 or args.worker_queue_episodes < 0 or args.milestone_every < 0:
        parser.error("Microbatch size, worker queue and milestone interval cannot be negative")
    args.ports = args.ports or [args.port]
    if len(set(args.ports)) != len(args.ports):
        parser.error("Server ports must be distinct")
    try:
        args.worker_ports = assign_ports(args.workers, args.ports)
    except ValueError as exc:
        parser.error(str(exc))
    if args.no_resume and args.resume_from:
        parser.error("--no-resume and --resume-from cannot be combined")
    return args


def select_checkpoint(args, checkpoint_dir: Path):
    """Returns (checkpoint to load, resume path for the learner).

    The newest learner_update_*.pt in the run directory wins; otherwise --resume-from;
    otherwise, outside imitation and smoke tests, the seed imitation checkpoint if present.
    """
    latest = sorted(checkpoint_dir.glob("learner_update_*.pt")) if not args.no_resume else []
    seed = default_seed_checkpoint()
    resume = args.resume_from
    if resume is None and args.mode != "imitation" and not args.no_resume and not args.model_config and seed.exists():
        resume = str(seed)
    selected = latest[-1] if latest else resume
    return selected, resume


def build_run_config(args, checkpoint_dir: Path, selected, resume) -> RunConfig:
    cfg = RunConfig.default()
    if args.model_config:
        architecture = ModelConfig(**json.loads(Path(args.model_config).read_text()))
    elif selected and not args.no_resume:
        architecture = model_config(read_checkpoint(selected))
    else:
        architecture = cfg.model
    return replace(cfg, model=architecture, matmul_precision=args.matmul_precision,
        infer=replace(cfg.infer, device=args.device),
        rollout=replace(cfg.rollout, target_concurrent_battles=args.concurrency, rooms_per_pair=args.concurrency,
            learn_max_pending_episodes=max(cfg.rollout.learn_max_pending_episodes, 2*args.concurrency)),
        learner=replace(cfg.learner, mode=args.mode, device=args.device, lr=args.learning_rate, lr_schedule=args.lr_schedule,
            minibatch_size=args.batch_size, update_epochs=args.epochs, steps_per_update=args.steps_per_update,
            microbatch_size=args.microbatch_size, kl_stop_rule=args.kl_stop_rule,
            ppo_lr_multipliers=args.lr_multipliers,
            max_total_steps=args.total_steps, max_updates=args.max_updates, resume=not args.no_resume,
            resume_path=str(Path(resume).resolve()) if resume else None,
            ckpt_dir=str(checkpoint_dir), save_every_updates=args.save_every,
            milestone_every_states=args.milestone_every))


def initial_status(args, cfg: RunConfig, checkpoint_dir: Path) -> dict:
    state = dict(status="starting", architecture="readme_15token", mode=args.mode,
        target_new_states=args.total_steps or None, checkpoint_dir=str(checkpoint_dir),
        model_config=cfg.as_dict()["model"], server_port=args.ports[0], server_ports=args.ports,
        worker_ports=args.worker_ports, recurrent=False, rollout_workers=args.workers,
        concurrent_battles=args.concurrency, worker_queue_episodes=args.worker_queue_episodes,
        matmul_precision=args.matmul_precision)
    state["learner_settings"] = dict(steps_per_update=args.steps_per_update, minibatch_size=args.batch_size,
        microbatch_size=args.microbatch_size, kl_stop_rule=args.kl_stop_rule, ppo_epochs=args.epochs,
        base_learning_rate=args.learning_rate, lr_schedule=args.lr_schedule,
        ppo_lr_multipliers=args.lr_multipliers, milestone_every_states=args.milestone_every)
    return state


async def main(argv=None):
    args = parse_args(argv)
    logs_dir().mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(logs_dir() / "training.log", encoding="utf-8")])
    checkpoint_dir = Path(args.checkpoint_dir or checkpoints_dir() / args.mode).resolve()
    selected, resume = select_checkpoint(args, checkpoint_dir)
    cfg = build_run_config(args, checkpoint_dir, selected, resume)
    state = initial_status(args, cfg, checkpoint_dir)
    started = time.monotonic()

    def save_status(**updates):
        state.update(updates, elapsed_seconds=time.monotonic()-started)
        atomic_json(args.status_file, state)

    save_status()
    try:
        gpu = args.device.startswith("cuda")
        init_ray(args.workers, gpu)
        _, infer, learner, workers = await start_actors(cfg, args, gpu)
        await monitor(learner, infer, workers, save_status, state, args.stop_file)
    except BaseException as exc:
        save_status(status="failed", error=repr(exc))
        raise
    finally:
        ray.shutdown()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
