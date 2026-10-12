from dataclasses import replace

import numpy as np
import pytest

from simengine.control import QueueController, QueuePolicy
from simengine.kernel import Simulation
from simengine.resource import Resource
from simengine.rng import Streams
from simengine.roster import shift
from simengine.schedule import synthetic_schedule
from simengine.terminal import TerminalConfig, generate_demand, simulate_terminal, terminal_kpis

POLICY = QueuePolicy(open_above=6.0, close_below=2.0, review_every=5.0, open_delay=5.0)
BASE = TerminalConfig(
    schedule=synthetic_schedule(),
    checkin_desks=shift(22, 3, 21),
    security_lanes=shift(12, 3, 21),
    security_policy=POLICY,
)


def surge(controlled: bool):
    """300 arrivals in the first hour at a resource that starts with one server."""
    sim = Simulation()
    r = Resource(sim, "r", 1)
    for i, t in enumerate(np.linspace(0, 60, 300)):
        sim.schedule_at(float(t), r.submit, i, 1.0)
    if controlled:
        QueueController(sim, r, POLICY, 1.0, lambda t: 10, stop_after=60.0)
    sim.run()
    return sim, r


def test_thresholds_must_not_overlap():
    with pytest.raises(ValueError):
        QueuePolicy(open_above=3.0, close_below=3.0)


def test_controller_opens_under_a_surge_and_closes_after():
    _, fixed = surge(controlled=False)
    sim, r = surge(controlled=True)
    lanes = [n for _, n in r.capacity_log]
    assert max(lanes) > 1 and max(lanes) <= 10
    assert r.capacity == POLICY.min_servers
    assert np.mean([v.wait for v in r.visits]) < 0.5 * np.mean([v.wait for v in fixed.visits])
    assert r.littles_law_residual() < 1e-9


def test_opening_takes_the_delay():
    sim, r = surge(controlled=True)
    first_open = next(t for t, n in r.capacity_log if n > 1)
    assert first_open >= POLICY.open_delay


@pytest.fixture(scope="module")
def day():
    demand = generate_demand(BASE, Streams(1, 0))
    run = simulate_terminal(BASE, demand)
    return demand, run, terminal_kpis(BASE, demand, run)


def test_lanes_stay_within_what_is_available(day):
    _, run, _ = day
    for t, n in run.security.capacity_log:
        assert POLICY.min_servers <= n or n == 0
        assert n <= BASE.security_lanes[int(t // 60) % 24] or n <= POLICY.min_servers


def test_everyone_clears_and_bookkeeping_holds(day):
    demand, run, kpis = day
    assert all(v.depart is not None for v in run.security.visits)
    assert run.security.littles_law_residual() < 1e-9
    # Lane hours also count a closing lane while it finishes its last passenger, the hourly lane
    # counts do not, so lane hours can only be slightly higher.
    hourly = sum(kpis[f"security_lanes_h{h:02d}"] for h in range(24))
    assert hourly <= kpis["security_lane_hours"] <= hourly + 0.1


def test_lower_threshold_means_more_lanes_and_shorter_waits(day):
    _, _, loose = day
    tight_config = replace(BASE, security_policy=replace(POLICY, open_above=2.0, close_below=0.5))
    tight = terminal_kpis(tight_config, generate_demand(tight_config, Streams(1, 0)),
                          simulate_terminal(tight_config, generate_demand(tight_config, Streams(1, 0))))
    assert tight["security_wait_mean"] < loose["security_wait_mean"]
    assert tight["security_lane_hours"] > loose["security_lane_hours"]
