"""Peaky arrivals against the M(t)/G/infinity result.

With unlimited lanes nobody queues, so the number at security at time t has a closed form (see
`seat_occupancy`). The engine has to reproduce its mean at every time of day, its variance, and
its full distribution at the peak.

Counts at different times of one replication are correlated, because they share each flight's
passenger count. So the mean check bounds each time separately with a Bonferroni correction
instead of summing a chi-square over times. Replications are independent, which is what the
distribution check at a single time relies on.
"""

import numpy as np
import pytest
from scipy import stats

from simengine.rng import Streams
from simengine.schedule import synthetic_schedule
from simengine.terminal import (
    ServiceTime,
    TerminalConfig,
    generate_demand,
    seat_occupancy,
    simulate_terminal,
)

REPS = 300
TIMES = np.arange(0, 24 * 60, 15.0)
CONFIG = TerminalConfig(
    schedule=synthetic_schedule(flights=20),
    checkin_desks=1,
    security_lanes=100_000,
    checkin_share=0.0,
    # A long, variable stay makes the occupancy large enough to test sharply.
    security_time=ServiceTime(mean=20.0, cv=0.8),
)


@pytest.fixture(scope="module")
def theory():
    p = seat_occupancy(CONFIG, TIMES)
    seats = np.array([f.seats for f in CONFIG.schedule])
    return p, seats @ p, seats @ (p * (1 - p))


@pytest.fixture(scope="module")
def counts():
    """Number at security at each time, per replication, read from the engine's visit records."""
    out = []
    for r in range(REPS):
        run = simulate_terminal(CONFIG, generate_demand(CONFIG, Streams(77, r)))
        arrive = np.array([v.arrive for v in run.security.visits])[:, None]
        depart = np.array([v.depart for v in run.security.visits])[:, None]
        out.append(((arrive <= TIMES) & (TIMES < depart)).sum(axis=0))
    return np.array(out)


def test_nobody_queues_with_unlimited_lanes():
    run = simulate_terminal(CONFIG, generate_demand(CONFIG, Streams(77, 0)))
    assert max(v.wait for v in run.security.visits) == 0


def test_mean_occupancy_matches_through_the_day(theory, counts):
    _, mean, var = theory
    busy = mean > 5
    z = (counts.mean(axis=0)[busy] - mean[busy]) / np.sqrt(var[busy] / REPS)
    bound = stats.norm.ppf(1 - 0.001 / (2 * busy.sum()))
    assert np.abs(z).max() < bound, f"worst time is {np.abs(z).max():.2f} standard errors out"
    # The test has teeth: the standard error at the peak is under 2% of the mean.
    peak = mean.argmax()
    assert np.sqrt(var[peak] / REPS) / mean[peak] < 0.02


def test_variance_matches_and_is_not_poisson(theory, counts):
    _, mean, var = theory
    busy = mean > 5
    observed = counts.var(axis=0, ddof=1)[busy].sum()
    assert observed / var[busy].sum() == pytest.approx(1.0, abs=0.05)
    # Binomial flight loads make the count under-dispersed. Poisson would overstate the variance.
    assert observed / mean[busy].sum() < 0.92


def test_distribution_at_the_peak(theory, counts):
    p, mean, _ = theory
    peak = mean.argmax()
    pmf = np.array([1.0])
    for seats, prob in zip((f.seats for f in CONFIG.schedule), p[:, peak]):
        pmf = np.convolve(pmf, stats.binom.pmf(np.arange(seats + 1), seats, prob))

    assert chi_square_pvalue(counts[:, peak], pmf) > 0.001


def chi_square_pvalue(sample: np.ndarray, pmf: np.ndarray, bins: int = 20) -> float:
    """Goodness of fit on quantile bins of `pmf`, so each bin expects roughly len(sample) / bins."""
    cdf = np.cumsum(pmf)
    edges = np.unique(np.searchsorted(cdf, np.linspace(0, 1, bins + 1)[1:-1]))
    expected = np.diff(np.concatenate([[0.0], cdf[edges], [1.0]])) * len(sample)
    observed = np.bincount(np.searchsorted(edges, sample), minlength=len(edges) + 1)
    return stats.chisquare(observed, expected).pvalue
