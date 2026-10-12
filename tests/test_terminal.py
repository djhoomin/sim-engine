from dataclasses import replace

import numpy as np
import pytest

from simengine.experiment import paired_difference, run_experiment
from simengine.rng import Streams
from simengine.schedule import Flight, load_schedule, synthetic_schedule
from simengine.terminal import TerminalConfig, generate_demand, simulate_terminal, terminal_model

SCHEDULE = synthetic_schedule()
BASE = TerminalConfig(schedule=SCHEDULE, checkin_desks=22, security_lanes=7)


@pytest.fixture(scope="module")
def day():
    demand = generate_demand(BASE, Streams(1, 0))
    return demand, simulate_terminal(BASE, demand)


def test_synthetic_schedule_is_fixed():
    assert synthetic_schedule() == synthetic_schedule()


def test_load_schedule_reads_csv(tmp_path):
    path = tmp_path / "schedule.csv"
    path.write_text("departure,seats\n18:05,186\n06:30,150\n")
    assert load_schedule(path) == (Flight(390.0, 150), Flight(1085.0, 186))


def test_demand_ignores_staffing():
    a = generate_demand(BASE, Streams(3, 2))
    b = generate_demand(replace(BASE, checkin_desks=5, security_lanes=20), Streams(3, 2))
    for name in ("flight", "arrival", "checks_in", "checkin_time", "security_time"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_check_in_share_only_moves_the_choice():
    a = generate_demand(BASE, Streams(3, 2))
    b = generate_demand(replace(BASE, checkin_share=0.6), Streams(3, 2))
    np.testing.assert_array_equal(a.arrival, b.arrival)
    np.testing.assert_array_equal(a.security_time, b.security_time)
    assert b.checks_in.sum() > a.checks_in.sum()


def test_every_passenger_clears_and_flow_is_conserved(day):
    demand, run = day
    assert len(run.checkin.visits) == demand.checks_in.sum()
    assert len(run.security.visits) == len(demand)
    assert all(v.depart is not None for v in run.checkin.visits + run.security.visits)
    assert run.sim.pending == 0


def test_arrivals_are_peaky(day):
    demand, _ = day
    hourly = np.histogram(demand.arrival, bins=np.arange(0, 24 * 60 + 1, 60))[0]
    busy = hourly[hourly > 0]
    assert busy.max() > 3 * busy.mean() / 2


def test_queues_are_first_come_first_served(day):
    _, run = day
    for resource in (run.checkin, run.security):
        starts = np.array([v.start for v in resource.visits])
        assert np.all(np.diff(starts) >= 0)


def test_littles_law_exact_on_both_resources(day):
    _, run = day
    assert run.checkin.littles_law_residual() < 1e-9
    assert run.security.littles_law_residual() < 1e-9


def test_unlimited_capacity_means_no_waiting():
    config = replace(BASE, checkin_desks=100_000, security_lanes=100_000)
    demand = generate_demand(config, Streams(4, 0))
    run = simulate_terminal(config, demand)
    security = sorted(run.security.visits, key=lambda v: v.entity)
    airside = np.array([v.depart for v in security])
    expected = demand.arrival + demand.checks_in * demand.checkin_time + demand.security_time
    np.testing.assert_allclose(airside, expected, rtol=1e-12, atol=1e-9)


def test_extra_security_lane_is_resolved_with_common_random_numbers():
    reps = 10
    seven = run_experiment(terminal_model, BASE, reps, base_seed=21)
    eight = run_experiment(terminal_model, replace(BASE, security_lanes=8), reps, base_seed=21)
    eight_indep = run_experiment(terminal_model, replace(BASE, security_lanes=8), reps, base_seed=22)

    assert paired_difference(seven, eight, "security_wait_p95").high < 0
    crn_sd = np.std(eight.kpi("security_wait_p95") - seven.kpi("security_wait_p95"), ddof=1)
    indep_sd = np.std(eight_indep.kpi("security_wait_p95") - seven.kpi("security_wait_p95"), ddof=1)
    assert crn_sd < indep_sd


def test_everyone_straight_to_security():
    config = replace(BASE, checkin_share=0.0)
    kpis = terminal_model(config, Streams(6, 0))
    assert kpis["checkin_wait_mean"] == 0.0
    assert kpis["passengers"] == sum(kpis[f"security_screened_h{h:02d}"] for h in range(24))


def test_transit_bunches_arrivals_without_making_anyone_late():
    from simengine.terminal import TransitLine

    rail = (TransitLine("metro", 0.27, headway=5.0), TransitLine("train", 0.33, headway=15.0, offset=7.0))
    plain = generate_demand(BASE, Streams(8, 0))
    bunched = generate_demand(replace(BASE, transit=rail), Streams(8, 0))
    assert len(bunched) == len(plain)
    # Same passengers and intended times, so rail can only bring people earlier, by under a headway
    # plus the walk.
    shift = np.sort(plain.arrival) - np.sort(bunched.arrival)
    assert shift.min() >= -1e-9

    def pulse_variance(arrivals: np.ndarray) -> float:
        # Variation of minute counts around a 15 minute moving average, leaving out the daily curve.
        per_minute = np.histogram(arrivals, np.arange(0, 1440, 1.0))[0].astype(float)
        trend = np.convolve(per_minute, np.ones(15) / 15, mode="same")
        return float(((per_minute - trend) ** 2).mean())

    assert pulse_variance(bunched.arrival) > 2 * pulse_variance(plain.arrival)


def test_transit_leaves_other_draws_alone():
    from simengine.terminal import TransitLine

    rail = (TransitLine("metro", 0.5, headway=5.0),)
    plain = generate_demand(BASE, Streams(9, 0))
    bunched = generate_demand(replace(BASE, transit=rail), Streams(9, 0))
    assert sorted(plain.security_time) == sorted(bunched.security_time)
