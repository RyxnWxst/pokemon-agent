"""The weight store: the learner publishes weights here, and the inference actor reads them."""
import ray


@ray.remote
class WeightStore:
    def __init__(self):
        self.weights = None
        self.version = -1

    def update(self, weights, version):
        self.weights, self.version = weights, version

    def get_version(self):
        return self.version

    def get_weights(self):
        return self.weights
