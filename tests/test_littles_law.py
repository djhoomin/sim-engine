import numpy as np
import pytest

from simengine.experiment import run_experiment
from simengine.station import MMcConfig, mmc_model, simulate_station


def test_exact_on_a_run_from_empty_to_empty():
    # When the window starts and ends empty, L = (N/T) * W holds exactly on the sample path,
    # whatever the distributions. Here with peaky gamma gaps and lognormal services on two servers.
    rng = np.random.default_rng(5)
    n = 3_000
    gaps = rng.gamma(shape=0.5, scale=1.6, size=n)
    station = simulate_station(
        iter([*gaps, np.inf]),
        iter(rng.lognormal(0.3, 0.8, size=n)),
        servers=2,
        horizon=float(gaps.sum()) + 1.0,
        drain=True,
    )
    assert len(station.visits) == n
    T = station.sim.now
    W = np.mean([v.sojourn for v in station.visits])
    L = station.in_system.mean(T)
    assert L == pytest.approx(n / T * W, rel=1e-9)
    assert station.littles_law_residual() < 1e-9


def test_bookkeeping_identity_holds_in_warmed_up_windows():
    config = MMcConfig(arrival_rate=2.4, service_rate=1.0, servers=3, horizon=2_000, warmup=300)
    record = run_experiment(mmc_model, config, replications=5, base_seed=1)
    assert record.kpi("littles_residual").max() < 1e-9


def test_holds_statistically_in_steady_state():
    config = MMcConfig(arrival_rate=0.7, service_rate=1.0, servers=1, horizon=20_000, warmup=2_000)
    record = run_experiment(mmc_model, config, replications=5, base_seed=2)
    L = record.kpi("L")
    lam_W = record.kpi("arrival_rate") * record.kpi("W")
    np.testing.assert_allclose(L, lam_W, rtol=0.02)
