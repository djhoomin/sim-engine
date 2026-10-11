"""The calibration code must recover parameters it is given before it is trusted on real data."""

import numpy as np
import pytest

from simengine.calibration import (
    GroupedShowUpFit,
    ShowUpFit,
    expected_showups,
    fit_dispersion,
    fit_showup,
    fit_showup_grouped,
)
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


def simulate_days(days: int, day_sd: float, concentration: float | None, rng) -> tuple:
    """Hourly counts for `days` copies of a 100-flight day, with optional day and flight variation."""
    day = synthetic_schedule(flights=100)
    departures = np.concatenate([[f.departure + 1440 * d for f in day] for d in range(days)])
    seats = np.tile([f.seats for f in day], days).astype(float)
    edges = np.arange(0, days * 1440 + 1, 60.0)
    p = np.full(len(departures), 0.8)
    if day_sd:
        p = p * np.repeat(np.exp(rng.normal(-day_sd**2 / 2, day_sd, days)), len(day))
    p = np.clip(p, 0, 0.999)
    if concentration:
        p = rng.beta(concentration * p, concentration * (1 - p))
    pax = rng.binomial(seats.astype(int), p)
    shape = (90 / 40) ** 2
    arrivals = np.repeat(departures, pax) - rng.gamma(shape, 90 / shape, pax.sum())
    counts = np.histogram(arrivals, edges)[0].astype(float)
    truth = GroupedShowUpFit((ShowUpFit(90.0, 40.0, 0.8, np.nan, len(counts)),), np.nan, len(counts))
    groups = np.zeros(len(departures), dtype=int)
    return departures, seats, groups, edges, counts, np.repeat(np.arange(days), 24), truth


def test_recovers_day_and_flight_variation():
    # 60 days so the day-level spread is estimable. The mean model is given, to test only this step.
    found = fit_dispersion(*simulate_days(60, 0.08, 15.0, Streams(3, 0).stream("cal")))
    assert found.day_sd == pytest.approx(0.08, rel=0.25)
    assert found.flight_concentration == pytest.approx(15.0, rel=0.25)


def test_no_extra_variation_gives_high_concentration():
    # Independent passengers only. The implied flight load factor sd should be close to zero.
    found = fit_dispersion(*simulate_days(60, 0.0, None, Streams(4, 0).stream("cal")))
    assert found.flight_concentration > 100


def test_timing_noise_is_not_mistaken_for_load_variation():
    # Shuffle a large share of passengers across neighbouring hours. Load variation cannot produce
    # that much hourly noise, so the estimator should say it is not identifiable.
    departures, seats, groups, edges, counts, day_of_bin, truth = simulate_days(30, 0.0, None, Streams(5, 0).stream("cal"))
    rng = Streams(5, 1).stream("cal")
    moved = (rng.random(len(counts)) - 0.5) * 0.6 * counts
    counts = counts - moved + np.roll(moved, 1)
    found = fit_dispersion(departures, seats, groups, edges, counts, day_of_bin, truth)
    assert np.isnan(found.flight_concentration)
