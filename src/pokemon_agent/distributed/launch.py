"""Start the Ray actors for a training run and watch them until it ends."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

import ray

from pokemon_agent.distributed.inference import InferenceActor
from pokemon_agent.distributed.learner import LearnerActor
from pokemon_agent.distributed.weights import WeightStore
from pokemon_agent.distributed.worker import RolloutWorker

ROLLOUT_KEYS = ("active_battles_worker", "learner_q_size", "traj_in_memory", "ep_sem_value",
    "model_games", "model_wins", "process_cpu_cores_average", "pending_message_tasks", "finished_rooms_tracked")
STALL_TIMEOUT_S = 300
POLL_INTERVAL_S = 2.


def init_ray(workers: int, gpu: bool) -> None:
    ray.init(num_cpus=max(4, workers+2), num_gpus=1 if gpu else 0, include_dashboard=False,
        object_store_memory=1024**3)


def worker_config(cfg, index: int, workers: int, concurrency: int, queue_episodes: int):
    """Split the concurrent battles evenly; earlier workers take the remainder."""
    battles = concurrency//workers + (index < concurrency % workers)
    return replace(cfg, rollout=replace(cfg.rollout, rooms_per_pair=battles,
        learn_max_pending_episodes=max(32, 2*battles), learn_queue_episodes=queue_episodes))


async def start_actors(cfg, args, gpu: bool):
    """Weight store, inference (0.35 GPU), learner (0.65 GPU), then one rollout worker per port."""
    store = WeightStore.remote()
    infer = ray.remote(num_gpus=.35 if gpu else 0)(InferenceActor).remote(cfg, store)
    learner = ray.remote(num_gpus=.65 if gpu else 0)(LearnerActor).remote(cfg, store)
    await asyncio.to_thread(ray.get, learner.ready.remote())
    worker_type = ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0)(RolloutWorker)
    workers = [worker_type.remote(worker_config(cfg, index, args.workers, args.concurrency, args.worker_queue_episodes),
        infer, learner, 1, args.worker_ports[index]) for index in range(args.workers)]
    return store, infer, learner, workers


def aggregate_rollout_stats(worker_statistics):
    rollout = {key: sum(worker.get(key, 0) for worker in worker_statistics) for key in ROLLOUT_KEYS}
    rollout["workers"] = len(worker_statistics)
    return rollout


async def monitor(learner, infer, workers, save_status, state, stop_file):
    """Poll every 2 s; save the status, stop on the stop file or completion, fail on a stall."""
    running = [worker.run.remote() for worker in workers]
    last_steps, last_progress = -1, time.monotonic()
    while True:
        finished, _ = ray.wait(running, timeout=0)
        if finished:
            ray.get(finished[0])
            raise RuntimeError("Rollout worker stopped unexpectedly")
        statistics = await asyncio.to_thread(ray.get,
            [learner.get_stats.remote(), infer.get_stats.remote(), *[worker.heartbeat.remote() for worker in workers]])
        training, inference = statistics[:2]
        worker_statistics = statistics[2:]
        save_status(status="running", learner=training, inference=inference,
            rollout=aggregate_rollout_stats(worker_statistics), worker_statistics=worker_statistics)
        if training["total_steps"] != last_steps:
            last_steps, last_progress = training["total_steps"], time.monotonic()
        elif time.monotonic()-last_progress > STALL_TIMEOUT_S:
            raise RuntimeError("No completed battle data received for five minutes")
        stop_file = Path(stop_file)
        requested = stop_file.is_file()
        if requested or training["completed"]:
            checkpoint = await asyncio.to_thread(ray.get, learner.save_now.remote())
            if requested:
                stop_file.unlink()
            save_status(status="stopped" if requested else "complete", checkpoint=checkpoint)
            print(json.dumps(state), flush=True)
            return checkpoint
        await asyncio.sleep(POLL_INTERVAL_S)
