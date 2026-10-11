from dataclasses import replace

import numpy as np
import pytest

from simengine.kernel import Simulation
from simengine.resource import Resource
from simengine.rng import Streams
from simengine.roster import as_roster, server_hours, shift, staff_to_load
from simengine.schedule import synthetic_schedule
from simengine.terminal import (
    TerminalConfig,
    expected_arrivals,
    expected_arrivals_by_hour,
    offered_load,
    generate_demand,
    simulate_terminal,
    terminal_kpis,
)

BASE = TerminalConfig(
    schedule=synthetic_schedule(), checkin_desks=shift(22, 3, 21), security_lanes=shift(7, 3, 21)
)


def test_roster_helpers():
    assert as_roster(3) == (3,) * 24
    assert shift(5, 6, 8) == (0,) * 6 + (5, 5) + (0,) * 16
    assert server_hours(shift(5, 6, 8)) == 10
    with pytest.raises(ValueError):
        as_roster((1, 2))
    load = [0.0] * 3 + [2.1] + [0.0] * 20
    assert staff_to_load(load, 0.7, minimum=1, open_hours=(2, 5))[1:5] == (0, 1, 3, 1)


def test_sub_hour_load_sizes_each_hour_to_its_busiest_interval():
    # Hour 3 averages 2.0 busy lanes but peaks at 3.6 in its first quarter.
    load = [0.0] * 96
    load[12:16] = [3.6, 2.4, 1.2, 0.8]
    assert np.mean(load[12:16]) == pytest.approx(2.0)
    assert staff_to_load(load, 0.9, minimum=0)[3] == 4
    assert staff_to_load(np.reshape(load, (24, 4)).mean(axis=1), 0.9, minimum=0)[3] == 3
    with pytest.raises(ValueError):
        staff_to_load([1.0] * 30, 0.9)


def test_opening_servers_starts_queued_work():
    sim = Simulation()
    r = Resource(sim, "r", 0)
    for i in range(3):
        r.submit(i, 1.0)
    assert r.busy == 0 and len(r.queue) == 3
    sim.schedule(2.0, r.set_capacity, 2)
    sim.run()
    assert [v.start for v in r.visits] == [2.0, 2.0, 3.0]


def test_closing_server_finishes_current_entity_first():
    sim = Simulation()
    r = Resource(sim, "r", 2)
    for i in range(3):
        r.submit(i, 4.0)
    sim.schedule(1.0, r.set_capacity, 1)
    sim.run()
    # Both in service finish at 4. Only one server stays open, so the third starts at 4 and ends 8.
    assert [v.depart for v in r.visits] == [4.0, 4.0, 8.0]
    assert r.littles_law_residual() < 1e-12
    # The closing server stays open until its entity leaves at 4: 2 servers to 4, then 1 to 8.
    assert r.server_time() == pytest.approx(12.0)
    assert r.utilisation() == pytest.approx(1.0)


def test_sub_hour_expected_arrivals_add_up_to_hourly():
    fine = expected_arrivals(BASE, 10.0)
    np.testing.assert_allclose(fine.reshape(24, 6).sum(axis=1), expected_arrivals_by_hour(BASE))
    loads = offered_load(BASE, 10.0)["security"]
    np.testing.assert_allclose(loads.reshape(24, 6).mean(axis=1), offered_load(BASE)["security"])


def test_expected_arrivals_match_simulated_average():
    expected = expected_arrivals_by_hour(BASE)
    simulated = np.mean(
        [
            np.histogram(generate_demand(BASE, Streams(5, r)).arrival, bins=60 * np.arange(25))[0]
            for r in range(200)
        ],
        axis=0,
    )
    busy = expected > 50
    np.testing.assert_allclose(simulated[busy], expected[busy], rtol=0.02)


@pytest.fixture(scope="module")
def day():
    demand = generate_demand(BASE, Streams(1, 0))
    run = simulate_terminal(BASE, demand)
    return demand, run, terminal_kpis(BASE, demand, run)


def test_lanes_follow_the_roster(day):
    _, run, kpis = day
    assert kpis["security_lane_hours"] == pytest.approx(server_hours(BASE.security_lanes))
    assert run.security.littles_law_residual() < 1e-9


def test_nobody_is_served_while_closed(day):
    _, run, _ = day
    starts = np.array([v.start for v in run.security.visits])
    assert starts.min() >= 3 * 60
    assert np.all(np.array([v.depart for v in run.security.visits]) <= 21 * 60)


def test_hourly_queue_accounts_for_every_minute_waited(day):
    _, run, kpis = day
    hourly_area = 60 * sum(kpis[f"security_queue_h{h:02d}"] for h in range(24))
    total_wait = sum(v.wait for v in run.security.visits)
    assert hourly_area == pytest.approx(total_wait, rel=1e-9)


def test_roster_that_closes_on_a_queue_is_an_error():
    config = replace(BASE, security_lanes=shift(7, 3, 10))
    with pytest.raises(RuntimeError, match="never cleared"):
        simulate_terminal(config, generate_demand(config, Streams(1, 0)))
