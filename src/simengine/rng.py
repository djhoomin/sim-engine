"""Deterministic named random streams for common random numbers.

Every source of randomness in a model draws from its own named stream. A stream is fixed by
the base seed, the replication index and the stream name, and by nothing else. Scenario
parameters never enter the seed, so two scenarios run with the same base seed see the same
arrivals and the same service requirements. That is what makes their difference a low-variance
comparison instead of noise.
"""

import hashlib

import numpy as np


def _name_key(name: str) -> int:
    # Python's hash() is salted per process, so derive a stable integer from the name instead.
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


class Streams:
    """Factory for the named streams of one replication."""

    def __init__(self, base_seed: int, replication: int):
        self.base_seed = base_seed
        self.replication = replication
        self._cache: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._cache:
            seq = np.random.SeedSequence(
                entropy=self.base_seed, spawn_key=(self.replication, _name_key(name))
            )
            self._cache[name] = np.random.Generator(np.random.PCG64(seq))
        return self._cache[name]
