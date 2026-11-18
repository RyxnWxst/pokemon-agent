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
from checkpoint_io import atomic_json, model_config, read_checkpoint
from config import ROOT, LearnerConfig, ModelConfig, RunConfig
from inference import InferenceActor, WeightStore
from learner import LearnerActor
from worker import RolloutWorker


def assign_ports(workers, ports):
    """Spread rollout workers across Showdown servers, round-robin."""
    if workers < len(ports):
        raise ValueError("Use at least one rollout worker per Showdown server")
    return [ports[index % len(ports)] for index in range(workers)]


def parse_args(argv=None):
    defaults = LearnerConfig()
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--status-file", default=str(ROOT / "logs/status.json"))
    parser.add_argument("--stop-file", default=str(ROOT / "logs/stop-training"))
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


async def main():
    args = parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "logs/training.log", encoding="utf-8")])
    checkpoint_dir = Path(args.checkpoint_dir or ROOT / "checkpoints" / args.mode).resolve()
    latest = sorted(checkpoint_dir.glob("learner_update_*.pt")) if not args.no_resume else []
    seed = ROOT / "checkpoints/seed/imitation-best.pt"
    resume = args.resume_from
    if resume is None and args.mode != "imitation" and not args.no_resume and not args.model_config and seed.exists():
        resume = str(seed)
    selected = latest[-1] if latest else resume
    cfg = RunConfig.default()
    if args.model_config:
        architecture = ModelConfig(**json.loads(Path(args.model_config).read_text()))
    elif selected and not args.no_resume:
        architecture = model_config(read_checkpoint(selected))
    else:
        architecture = cfg.model
    cfg = replace(cfg, model=architecture, matmul_precision=args.matmul_precision,
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
    started = last_progress = time.monotonic()
    last_steps = -1

    def save_status(**updates):
        state.update(updates, elapsed_seconds=time.monotonic()-started)
        atomic_json(args.status_file, state)

    save_status()
    try:
        gpu = args.device.startswith("cuda")
        ray.init(num_cpus=max(4, args.workers+2), num_gpus=1 if gpu else 0, include_dashboard=False,
            object_store_memory=1024**3)
        store = WeightStore.remote()
        infer = ray.remote(num_gpus=.35 if gpu else 0)(InferenceActor).remote(cfg, store)
        learner = ray.remote(num_gpus=.65 if gpu else 0)(LearnerActor).remote(cfg, store)
        await asyncio.to_thread(ray.get, learner.ready.remote())
        worker_type = ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0)(RolloutWorker)
        workers = []
        for index in range(args.workers):
            battles = args.concurrency//args.workers + (index < args.concurrency % args.workers)
            worker_cfg = replace(cfg, rollout=replace(cfg.rollout, rooms_per_pair=battles,
                learn_max_pending_episodes=max(32, 2*battles), learn_queue_episodes=args.worker_queue_episodes))
            workers.append(worker_type.remote(worker_cfg, infer, learner, 1, args.worker_ports[index]))
        running = [worker.run.remote() for worker in workers]
        while True:
            finished, _ = ray.wait(running, timeout=0)
            if finished:
                ray.get(finished[0])
                raise RuntimeError("Rollout worker stopped unexpectedly")
            statistics = await asyncio.to_thread(ray.get,
                [learner.get_stats.remote(), infer.get_stats.remote(), *[worker.heartbeat.remote() for worker in workers]])
            training, inference = statistics[:2]
            worker_statistics = statistics[2:]
            keys = ("active_battles_worker", "learner_q_size", "traj_in_memory", "ep_sem_value",
                    "teacher_labels", "student_actions", "teacher_actions", "model_games", "model_wins",
                    "process_cpu_cores_average", "pending_message_tasks", "finished_rooms_tracked")
            rollout = {key: sum(worker.get(key, 0) for worker in worker_statistics) for key in keys}
            rollout["workers"] = len(workers)
            save_status(status="running", learner=training, inference=inference, rollout=rollout,
                worker_statistics=worker_statistics)
            if training["total_steps"] != last_steps:
                last_steps, last_progress = training["total_steps"], time.monotonic()
            elif time.monotonic()-last_progress > 300:
                raise RuntimeError("No completed battle data received for five minutes")
            stop_file = Path(args.stop_file)
            requested = stop_file.is_file()
            if requested or training["completed"]:
                checkpoint = await asyncio.to_thread(ray.get, learner.save_now.remote())
                if requested:
                    stop_file.unlink()
                save_status(status="stopped" if requested else "complete", checkpoint=checkpoint)
                print(json.dumps(state), flush=True)
                break
            await asyncio.sleep(2.)
    except BaseException as exc:
        save_status(status="failed", error=repr(exc))
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
