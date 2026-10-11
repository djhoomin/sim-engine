import os
import subprocess
import sys

import numpy as np

from simengine.rng import Streams


def test_same_seed_same_draws():
    a = Streams(42, 0).stream("arrivals").random(5)
    b = Streams(42, 0).stream("arrivals").random(5)
    np.testing.assert_array_equal(a, b)


def test_names_replications_and_seeds_give_distinct_streams():
    base = Streams(42, 0).stream("arrivals").random(5)
    for other in (
        Streams(42, 0).stream("service"),
        Streams(42, 1).stream("arrivals"),
        Streams(43, 0).stream("arrivals"),
    ):
        assert not np.array_equal(base, other.random(5))


def test_stream_unaffected_by_other_streams():
    # Adding a new source of randomness to a model must not shift the existing ones.
    plain = Streams(7, 3).stream("arrivals").random(5)
    s = Streams(7, 3)
    s.stream("service").random(100)
    s.stream("new_thing").random(100)
    np.testing.assert_array_equal(plain, s.stream("arrivals").random(5))


def test_stable_across_processes():
    # Python's string hash is salted per process. Stream identity must not depend on it.
    code = "from simengine.rng import Streams; print(Streams(1, 2).stream('security').random())"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "1", "12345")
    }
    assert len(outputs) == 1
