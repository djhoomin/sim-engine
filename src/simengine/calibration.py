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


@dataclass(frozen=True)
class Dispersion:
    """Variation in demand beyond independent passengers.

    `day_sd` is the standard deviation of the log of a multiplier on every flight's load factor,
    one draw per day. `flight_concentration` is the beta concentration of each flight's own load
    factor around its mean, where smaller means more variation and infinity means none. It is NaN
    when the residuals exceed what any load variation could produce, even every flight being either
    full or empty, so that variation must come from something else, such as timing.
    """

    day_sd: float
    flight_concentration: float
    # Share of squared hourly residuals that repeats at the same hour every day. That part is an
    # error in the mean model's shape, which no amount of random variation represents.
    systematic_share: float = 0.0


def bin_probabilities(
    departures: np.ndarray, groups: np.ndarray, edges: np.ndarray, fit: "GroupedShowUpFit"
) -> tuple[np.ndarray, np.ndarray]:
    """Per flight, the chance one passenger lands in each bin, and the flight's mean load factor."""
    q = np.zeros((len(departures), len(edges) - 1))
    load = np.zeros(len(departures))
    for g, profile in enumerate(fit.groups):
        mine = groups == g
        shape, scale = gamma_shape_scale(profile.mean, profile.sd)
        cdf = stats.gamma.cdf(np.subtract.outer(departures[mine], edges), shape, scale=scale)
        q[mine] = cdf[:, :-1] - cdf[:, 1:]
        load[mine] = profile.load_factor
    return q, load


def fit_dispersion(
    departures: np.ndarray,
    seats: np.ndarray,
    groups: np.ndarray,
    edges: np.ndarray,
    observed: np.ndarray,
    day_of_bin: np.ndarray,
    fit: "GroupedShowUpFit",
    hour_of_bin: np.ndarray | None = None,
) -> Dispersion:
    """Method of moments estimates of day and flight variation, given a fitted mean model.

    Day: each day's observed total over its expected total is that day's multiplier, and the
    spread of their logs estimates `day_sd`. Flight: with each day's multiplier applied, the squared
    hourly residuals beyond what independent passengers explain are set equal to the extra variance
    a beta load factor adds, which for flight f in bin b is s(s - 1) q^2 p(1 - p) / (concentration + 1).

    With `hour_of_bin`, the residual that repeats at the same hour on every day is removed first and
    reported as `systematic_share`, so the flight term only takes variation from day to day. That
    needs the same bins on every day. Without it, the flight term takes all hourly variation the mean
    model does not explain, shape errors included.
    """
    departures, seats, groups, edges, observed, day_of_bin = map(
        np.asarray, (departures, seats, groups, edges, observed, day_of_bin)
    )
    q, load = bin_probabilities(departures, groups, edges, fit)
    expected = (seats * load) @ q
    days = np.unique(day_of_bin)
    ratio = np.array([observed[day_of_bin == d].sum() / expected[day_of_bin == d].sum() for d in days])
    day_sd = float(np.std(np.log(ratio), ddof=1)) if len(days) > 1 else 0.0

    flight_day = np.clip(np.floor(departures / 1440).astype(int), 0, len(days) - 1)
    p = np.clip(load * ratio[flight_day], 0.0, 0.999)[:, None]
    mean = (seats[:, None] * q * p).sum(axis=0)
    passenger_var = (seats[:, None] * q * p * (1 - q * p)).sum(axis=0)
    flight_unit = ((seats * (seats - 1))[:, None] * q**2 * p * (1 - p)).sum(axis=0)
    residual = observed - mean
    systematic_share = 0.0
    if hour_of_bin is not None and len(days) > 1:
        hours = np.asarray(hour_of_bin)
        by_hour = {h: residual[hours == h].mean() for h in np.unique(hours)}
        systematic = np.array([by_hour[h] for h in hours])
        systematic_share = float((systematic**2).sum() / (residual**2).sum())
        # Removing a mean per hour over D days takes away 1/D of the random variance as well.
        residual = (residual - systematic) * np.sqrt(len(days) / (len(days) - 1))
    excess = float((residual**2 - passenger_var).sum())
    if excess <= 0:
        concentration = np.inf
    elif excess >= flight_unit.sum():
        concentration = np.nan
    else:
        concentration = flight_unit.sum() / excess - 1
    return Dispersion(day_sd, float(concentration), systematic_share)


def fit_scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Error measures for predicted against observed counts."""
    err = np.asarray(predicted) - np.asarray(observed)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / np.sum((observed - np.mean(observed)) ** 2)),
        "total_error_pct": float(100 * err.sum() / np.sum(observed)),
    }
