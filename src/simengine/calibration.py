"""Fit the demand model to observed checkpoint counts.

The model behind the terminal's demand: each departure with s seats sends load_factor * s
passengers through security, each arriving a gamma distributed lead time before departure. Given
a schedule and observed counts per time bin, this fits the gamma mean and standard deviation and
the effective load factor by Poisson maximum likelihood.

Times are minutes on one continuous clock, so several days can be fitted together. Bins do not
have to be equal, which lets a closed checkpoint's hours fold into the bin when it reopens.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


def gamma_shape_scale(mean: float, sd: float) -> tuple[float, float]:
    shape = (mean / sd) ** 2
    return shape, mean / shape


def expected_showups(
    departures: np.ndarray, seats: np.ndarray, edges: np.ndarray, mean: float, sd: float
) -> np.ndarray:
    """Expected passengers per bin with a load factor of one.

    Arriving in [a, b) means a lead time in (departure - b, departure - a].
    """
    shape, scale = gamma_shape_scale(mean, sd)
    cdf = stats.gamma.cdf(np.subtract.outer(departures, edges), shape, scale=scale)
    return seats @ (cdf[:, :-1] - cdf[:, 1:])


@dataclass(frozen=True)
class ShowUpFit:
    mean: float
    sd: float
    load_factor: float
    log_likelihood: float
    bins: int

    def predict(self, departures: np.ndarray, seats: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return self.load_factor * expected_showups(departures, seats, edges, self.mean, self.sd)


def _poisson_loglik(observed: np.ndarray, expected: np.ndarray) -> float:
    expected = np.maximum(expected, 1e-12)
    return float(np.sum(observed * np.log(expected) - expected - special.gammaln(observed + 1)))


def fit_showup(
    departures: np.ndarray,
    seats: np.ndarray,
    edges: np.ndarray,
    observed: np.ndarray,
    start: tuple[float, float] = (110.0, 35.0),
) -> ShowUpFit:
    """Maximum likelihood show-up profile and load factor.

    For a fixed profile the best load factor is total observed over total expected, so the search
    runs over the gamma mean and standard deviation only, on a log scale to keep both positive.
    """
    departures, seats, edges, observed = map(np.asarray, (departures, seats, edges, observed))

    def profile(params: np.ndarray) -> tuple[float, float]:
        mean, sd = np.exp(params)
        unit = expected_showups(departures, seats, edges, mean, sd)
        load = observed.sum() / unit.sum()
        return load, _poisson_loglik(observed, load * unit)

    result = optimize.minimize(
        lambda p: -profile(p)[1], np.log(start), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6}
    )
    mean, sd = np.exp(result.x)
    load, loglik = profile(result.x)
    return ShowUpFit(float(mean), float(sd), float(load), loglik, len(observed))


@dataclass(frozen=True)
class GroupedShowUpFit:
    """One show-up profile and load factor per group of departures, such as by time of day."""

    groups: tuple[ShowUpFit, ...]
    log_likelihood: float
    bins: int

    def predict(
        self, departures: np.ndarray, seats: np.ndarray, groups: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        return sum(
            g.predict(departures[groups == k], seats[groups == k], edges)
            for k, g in enumerate(self.groups)
        )


def fit_showup_grouped(
    departures: np.ndarray,
    seats: np.ndarray,
    groups: np.ndarray,
    edges: np.ndarray,
    observed: np.ndarray,
) -> GroupedShowUpFit:
    """Maximum likelihood fit with a separate mean, sd and load factor for each group.

    Groups are numbered 0 to k - 1. Starts every group from the pooled fit, then searches all
    3k log parameters together, since groups overlap in time and their loads do not separate.
    """
    departures, seats, groups, edges, observed = map(np.asarray, (departures, seats, groups, edges, observed))
    k = int(groups.max()) + 1
    pooled = fit_showup(departures, seats, edges, observed)

    def expected(params: np.ndarray) -> np.ndarray:
        p = np.exp(params).reshape(k, 3)
        return sum(
            p[g, 2] * expected_showups(departures[groups == g], seats[groups == g], edges, p[g, 0], p[g, 1])
            for g in range(k)
        )

    start = np.log(np.tile([pooled.mean, pooled.sd, pooled.load_factor], k))
    result = optimize.minimize(
        lambda q: -_poisson_loglik(observed, expected(q)), start, method="L-BFGS-B"
    )
    p = np.exp(result.x).reshape(k, 3)
    loglik = _poisson_loglik(observed, expected(result.x))
    fits = tuple(ShowUpFit(float(m), float(s), float(l), float("nan"), len(observed)) for m, s, l in p)
    return GroupedShowUpFit(fits, loglik, len(observed))
