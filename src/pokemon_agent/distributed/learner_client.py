"""Ships finished episodes from a rollout worker to the learner on a background thread."""
import logging
import queue
import threading

import numpy as np
import ray

from pokemon_agent.config import RunConfig
from pokemon_agent.rl.buffer import pack_episodes

logger = logging.getLogger(__name__)


class SyncLearnerClient:
    def __init__(self, learner_actor, cfg: RunConfig):
        self.learner_actor = learner_actor
        self.cfg = cfg.rollout
        self.q = queue.Queue(maxsize=self.cfg.learn_queue_episodes
            or self.cfg.learn_max_pending_batches * self.cfg.learn_max_episodes)
        self.failure = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit_episode(self, obs: np.ndarray, act: np.ndarray, logp: np.ndarray,
                       val: np.ndarray, var: np.ndarray, rew: np.ndarray, done: np.ndarray) -> bool:
        # Apply backpressure rather than silently discarding completed games.
        if self.failure:
            raise RuntimeError(self.failure)
        self.q.put((obs, act, logp, val, var, rew, done))
        return True

    def _worker(self):
        while True:
            items = [self.q.get()]
            while len(items) < self.cfg.learn_max_episodes:
                try: items.append(self.q.get_nowait())
                except queue.Empty: break

            packed = pack_episodes(items)
            try:
                ray.get(self.learner_actor.submit_packed_batch.remote(*packed))
            except Exception as e:
                self.failure = f"Learner submission failed: {e!r}"
                logger.error(self.failure)
                return
