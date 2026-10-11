"""The engine against closed-form theory. These must pass before the model meets any real data.

Each check runs replicated experiments with a fixed seed, so the outcome is deterministic. A
check passes when the analytical value sits inside the 99% interval and that interval is tight
enough to mean something.
"""

import numpy as np
import pytest

from simengine.experiment import run_experiment
from simengine.queueing import lindley_waits, mm1, mmc
from simengine.station import MMcConfig, mmc_model, simulate_station

REPS = 20
LEVEL = 0.99
MAX_RELATIVE_HALF_WIDTH = 0.05


def assert_matches(record, kpi, expected):
    ci = record.interval(kpi, LEVEL)
    assert ci.contains(expected), f"{kpi}: theory {expected:.4f} outside {ci.low:.4f}..{ci.high:.4f}"
    assert ci.half_width / expected < MAX_RELATIVE_HALF_WIDTH, f"{kpi}: interval too wide to test"


@pytest.fixture(scope="module")
def mm1_run():
    config = MMcConfig(arrival_rate=0.7, service_rate=1.0, servers=1, horizon=40_000, warmup=2_000)
    return config, run_experiment(mmc_model, config, REPS, base_seed=20261011)


@pytest.fixture(scope="module")
def mmc_run():
    config = MMcConfig(arrival_rate=2.4, service_rate=1.0, servers=3, horizon=30_000, warmup=1_000)
    return config, run_experiment(mmc_model, config, REPS, base_seed=20261011)


@pytest.mark.parametrize("kpi", ["W", "Wq", "L", "Lq", "utilisation"])
def test_mm1(mm1_run, kpi):
    config, record = mm1_run
    theory = mm1(config.arrival_rate, config.service_rate)
    expected = {"utilisation": theory.rho}.get(kpi, getattr(theory, kpi, None))
    assert_matches(record, kpi, expected)


@pytest.mark.parametrize("kpi", ["W", "Wq", "L", "Lq", "utilisation"])
def test_mmc_erlang_c(mmc_run, kpi):
    config, record = mmc_run
    theory = mmc(config.arrival_rate, config.service_rate, config.servers)
    expected = {"utilisation": theory.rho}.get(kpi, getattr(theory, kpi, None))
    assert_matches(record, kpi, expected)


def test_lindley_recursion_matches_engine_exactly():
    # Single FIFO server with non-exponential gaps and services: the engine's waits must equal
    # the Lindley recursion customer by customer, not just on average.
    rng = np.random.default_rng(11)
    n = 5_000
    gaps = rng.gamma(shape=2.0, scale=0.6, size=n)
    services = rng.lognormal(mean=-0.2, sigma=0.5, size=n)
    station = simulate_station(
        iter([*gaps, np.inf]),
        iter(services),
        servers=1,
        horizon=float(gaps.sum()) + 1.0,  # margin so rounding cannot drop the last arrival
        drain=True,
    )
    engine = np.array([v.wait for v in sorted(station.visits, key=lambda v: v.entity)])
    assert len(engine) == n
    np.testing.assert_allclose(engine, lindley_waits(gaps, services), rtol=1e-9, atol=1e-8)
