"""The calibration code must recover parameters it is given before it is trusted on real data."""

import numpy as np
import pytest

from simengine.calibration import expected_showups, fit_showup, fit_showup_grouped
from simengine.rng import Streams
from simengine.schedule import synthetic_schedule

DAYS = 7
EDGES = np.arange(0, DAYS * 1440 + 1, 60.0)


def week_of_departures() -> tuple[np.ndarray, np.ndarray]:
    day = synthetic_schedule(flights=100)
    departures = np.concatenate([[f.departure + 1440 * d for f in day] for d in range(DAYS)])
    seats = np.tile([f.seats for f in day], DAYS).astype(float)
    return departures, seats


def simulate_counts(departures, seats, mean, sd, load, rng) -> np.ndarray:
    pax = rng.binomial(seats.astype(int), load)
    shape = (mean / sd) ** 2
    arrivals = np.repeat(departures, pax) - rng.gamma(shape, mean / shape, pax.sum())
    return np.histogram(arrivals, EDGES)[0].astype(float)


def test_expected_showups_matches_terminal_maths():
    departures = np.array([600.0])
    unit = expected_showups(departures, np.array([100.0]), np.array([0, 480, 540, 600, 1440.0]), 60, 30)
    assert unit.sum() == pytest.approx(100.0, rel=1e-6)
    assert unit[2] > unit[1] and unit[3] == pytest.approx(0.0)


def test_recovers_a_single_profile():
    departures, seats = week_of_departures()
    counts = simulate_counts(departures, seats, 95.0, 40.0, 0.82, Streams(1, 0).stream("cal"))
    fit = fit_showup(departures, seats, EDGES, counts)
    assert fit.mean == pytest.approx(95.0, rel=0.02)
    assert fit.sd == pytest.approx(40.0, rel=0.04)
    assert fit.load_factor == pytest.approx(0.82, rel=0.01)


def test_recovers_two_profiles_by_time_of_day():
    departures, seats = week_of_departures()
    group = (departures % 1440 >= 12 * 60).astype(int)
    rng = Streams(2, 0).stream("cal")
    counts = simulate_counts(departures[group == 0], seats[group == 0], 70.0, 30.0, 0.8, rng)
    counts += simulate_counts(departures[group == 1], seats[group == 1], 120.0, 50.0, 0.9, rng)
    fit = fit_showup_grouped(departures, seats, group, EDGES, counts)
    morning, afternoon = fit.groups
    assert (morning.mean, morning.sd, morning.load_factor) == pytest.approx((70, 30, 0.8), rel=0.05)
    assert (afternoon.mean, afternoon.sd, afternoon.load_factor) == pytest.approx((120, 50, 0.9), rel=0.05)
