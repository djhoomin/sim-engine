from dataclasses import replace

import numpy as np
import pytest

from simengine.checkpoint import Checkpoint, LaneGroup, Layout
from simengine.control import QueuePolicy
from simengine.kernel import Simulation
from simengine.rng import Streams
from simengine.roster import shift
from simengine.schedule import synthetic_schedule
from simengine.terminal import TerminalConfig, generate_demand, simulate_terminal, terminal_kpis

KASTRUP_2025 = Layout(
    (*(LaneGroup(f"ct{i}", 2, speed=0.75) for i in range(1, 6)), LaneGroup("ct6", 1, speed=0.75), LaneGroup("xray", 2))
)


def run(layout: Layout, capacity: int, arrivals: int = 400, spacing: float = 0.1):
    sim = Simulation()
    cp = Checkpoint(sim, layout, capacity, service_mean=1.0)
    for i in range(arrivals):
        sim.schedule_at(i * spacing, cp.submit, i, 1.0)
    sim.run()
    return sim, cp


def test_lanes_open_in_order_and_gates_skip_closed_groups():
    _, cp = run(KASTRUP_2025, capacity=4)
    assert cp.target == [2, 2, 0, 0, 0, 0, 0]
    assert set(cp.group_of.values()) == {"ct1", "ct2"}


def test_shortest_wait_routing_balances_equal_groups():
    layout = Layout((LaneGroup("a", 2), LaneGroup("b", 2)))
    _, cp = run(layout, capacity=4)
    share_a = sum(g == "a" for g in cp.group_of.values()) / len(cp.group_of)
    assert share_a == pytest.approx(0.5, abs=0.05)


def test_group_speed_scales_service_time():
    _, cp = run(Layout((LaneGroup("ct", 2, speed=0.75),)), capacity=2, arrivals=10)
    assert all(v.service_time == pytest.approx(0.75) for v in cp.visits)


def test_closing_group_drains_its_queue_first():
    sim = Simulation()
    cp = Checkpoint(sim, Layout((LaneGroup("a", 1), LaneGroup("b", 1))), 2, service_mean=1.0)
    for i in range(20):
        sim.schedule_at(0.0, cp.submit, i, 1.0)
    sim.schedule_at(0.5, cp.set_capacity, 1)
    sim.run()
    b = cp.groups[1]
    assert all(v.depart is not None for v in cp.visits)
    assert b.capacity == 0
    assert all(v.arrive <= 0.5 for v in b.visits)  # no new passengers after it was told to close


def test_one_group_matches_a_single_queue_exactly():
    base = TerminalConfig(schedule=synthetic_schedule(), checkin_desks=shift(22, 3, 21), security_lanes=shift(7, 3, 21))
    grouped = replace(base, security_layout=Layout((LaneGroup("all", 7),)))
    a = terminal_kpis(base, d := generate_demand(base, Streams(1, 0)), simulate_terminal(base, d))
    b = terminal_kpis(grouped, d, simulate_terminal(grouped, d))
    for k in ("security_wait_mean", "security_wait_p95", "security_queue_peak", "security_lane_hours"):
        assert b[k] == pytest.approx(a[k], rel=1e-12), k


def test_kastrup_layout_with_staffing_policy():
    config = TerminalConfig(
        schedule=synthetic_schedule(),
        checkin_desks=shift(22, 3, 21),
        security_lanes=10,  # staff for at most ten lanes
        security_layout=KASTRUP_2025,
        security_policy=QueuePolicy(open_above=6.0, close_below=2.0, min_servers=2),
    )
    demand = generate_demand(config, Streams(2, 0))
    result = simulate_terminal(config, demand)
    kpis = terminal_kpis(config, demand, result)
    assert all(v.depart is not None for v in result.security.visits)
    assert max(n for _, n in result.security.capacity_log) <= 10
    assert result.security.littles_law_residual() < 1e-9
    assert kpis["security_share_ct"] + kpis.get("security_share_xray", 0.0) == pytest.approx(1.0)
