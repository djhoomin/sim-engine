"""A single departures terminal: check-in feeding security.

Demand comes from the flight schedule. Each flight's passenger count is drawn from its seats and
load factor, and each passenger shows up a random lead time before departure. Summing those
show-up curves over the schedule gives the peaky, non-stationary arrival process that a constant
Poisson rate would miss.

All passengers of a replication are generated up front from their own named streams, before any
resource exists. Staffing never touches demand, so two staffing scenarios with the same seed see
the same passengers arriving at the same moments with the same service needs.

Desks and lanes follow hourly rosters. A desk or lane that closes finishes the passenger it is
serving first.

The day starts and ends empty. That makes this a terminating simulation with no warm-up to
choose, and it means Little's Law holds exactly on each resource.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .kernel import Simulation
from .resource import Resource, Visit
from .rng import Streams
from .roster import HOURS, Roster, as_roster
from .schedule import Flight


@dataclass(frozen=True)
class ServiceTime:
    """Lognormal service time given by its mean and coefficient of variation, in minutes."""

    mean: float
    cv: float

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        sigma2 = np.log1p(self.cv**2)
        return rng.lognormal(np.log(self.mean) - sigma2 / 2, np.sqrt(sigma2), n)


@dataclass(frozen=True)
class TerminalConfig:
    schedule: tuple[Flight, ...]
    checkin_desks: int | Roster
    security_lanes: int | Roster
    load_factor: float = 0.85
    checkin_share: float = 0.4  # passengers with bags to drop, the rest go straight to security
    showup_mean: float = 110.0  # minutes before departure
    showup_sd: float = 35.0
    checkin_time: ServiceTime = field(default_factory=lambda: ServiceTime(2.5, 0.6))
    security_time: ServiceTime = field(default_factory=lambda: ServiceTime(0.4, 0.5))
    security_cutoff: float = 30.0  # clearing security later than this before departure counts as late


@dataclass(frozen=True)
class Demand:
    """One replication's passengers, sorted by arrival. Index in these arrays is the entity id."""

    flight: np.ndarray
    arrival: np.ndarray
    checks_in: np.ndarray
    checkin_time: np.ndarray
    security_time: np.ndarray

    def __len__(self) -> int:
        return len(self.arrival)


def generate_demand(config: TerminalConfig, streams: Streams) -> Demand:
    departures = np.array([f.departure for f in config.schedule])
    seats = np.array([f.seats for f in config.schedule])
    pax = streams.stream("load").binomial(seats, config.load_factor)
    flight = np.repeat(np.arange(len(seats)), pax)
    n = len(flight)

    shape = (config.showup_mean / config.showup_sd) ** 2
    lead = streams.stream("showup").gamma(shape, config.showup_mean / shape, n)
    # Service times are drawn for every passenger, including those who skip check-in, so that
    # changing the check-in share does not shift anyone else's draws.
    order = np.argsort(departures[flight] - lead, kind="stable")
    return Demand(
        flight=flight[order],
        arrival=(departures[flight] - lead)[order],
        checks_in=(streams.stream("checkin_choice").random(n) < config.checkin_share)[order],
        checkin_time=config.checkin_time.sample(streams.stream("checkin_time"), n)[order],
        security_time=config.security_time.sample(streams.stream("security_time"), n)[order],
    )


def expected_arrivals(config: TerminalConfig, interval: float = 60.0) -> np.ndarray:
    """Expected passengers arriving in each `interval` minutes of the day, from the schedule."""
    if (24 * 60) % interval:
        raise ValueError("interval must divide the day evenly")
    shape = (config.showup_mean / config.showup_sd) ** 2
    lead = stats.gamma(shape, scale=config.showup_mean / shape)
    edges = np.arange(0, 24 * 60 + interval / 2, interval)
    expected = np.zeros(len(edges) - 1)
    for f in config.schedule:
        # Arriving in [a, b) means a lead time in (departure - b, departure - a].
        cdf = lead.cdf(f.departure - edges)
        expected += f.seats * config.load_factor * (cdf[:-1] - cdf[1:])
    return expected


def expected_arrivals_by_hour(config: TerminalConfig) -> np.ndarray:
    return expected_arrivals(config, 60.0)


def offered_load(config: TerminalConfig, interval: float = 60.0) -> dict[str, np.ndarray]:
    """Expected number of busy desks and lanes in each `interval` minutes of the day.

    Ignores the few minutes check-in adds before security, which is small next to an interval.
    """
    arrivals = expected_arrivals(config, interval)
    return {
        "checkin": arrivals * config.checkin_share * config.checkin_time.mean / interval,
        "security": arrivals * config.security_time.mean / interval,
    }


def seat_occupancy(config: TerminalConfig, times: np.ndarray, nodes: int = 64) -> np.ndarray:
    """Probability that a given seat's passenger is at security at each time, with unlimited lanes.

    Returns an array indexed [flight, time]. This is the M(t)/G/infinity result adapted to this
    model's demand. A seat is filled with probability equal to the load factor, its passenger
    arrives at departure minus a gamma lead time A, and stays for a service time S. With no queue
    the passenger is present at t exactly when A <= t < A + S, so the probability is
    load_factor * E_S[P(t - S < A <= t)], with the expectation over S by Gauss-Hermite quadrature.

    Seats are independent, so the number at security at time t is a sum of binomials with these
    probabilities. Its mean is the M(t)/G/infinity mean. Its variance is slightly below the Poisson
    variance, because each flight's passenger count is binomial rather than Poisson.

    Only valid when nobody goes through check-in first, so the check-in share must be zero.
    """
    if config.checkin_share != 0:
        raise ValueError("seat_occupancy assumes everyone goes straight to security")
    shape = (config.showup_mean / config.showup_sd) ** 2
    lead = stats.gamma(shape, scale=config.showup_mean / shape)
    sigma2 = np.log1p(config.security_time.cv**2)
    x, w = np.polynomial.hermite.hermgauss(nodes)
    service = np.exp(np.log(config.security_time.mean) - sigma2 / 2 + np.sqrt(2 * sigma2) * x)
    weights = w / np.sqrt(np.pi)

    departures = np.array([f.departure for f in config.schedule])[:, None, None]
    t = np.asarray(times, dtype=float)[None, :, None]
    # A <= t means lead >= departure - t, and A > t - S means lead < departure - t + S.
    present = lead.sf(departures - t) - lead.sf(departures - t + service[None, None, :])
    return config.load_factor * (present * weights).sum(axis=2)


@dataclass
class TerminalRun:
    sim: Simulation
    checkin: Resource
    security: Resource


def simulate_terminal(config: TerminalConfig, demand: Demand) -> TerminalRun:
    sim = Simulation(start=float(demand.arrival.min()))
    rosters = {"checkin": as_roster(config.checkin_desks), "security": as_roster(config.security_lanes)}
    first_hour = int(sim.now // 60)
    checkin = Resource(sim, "checkin", rosters["checkin"][first_hour % HOURS])
    security = Resource(sim, "security", rosters["security"][first_hour % HOURS])
    # Capacity changes on the hour through to the end of the day, then holds at the last count.
    for hour in range(first_hour + 1, HOURS):
        for resource in (checkin, security):
            sim.schedule_at(60.0 * hour, resource.set_capacity, rosters[resource.name][hour % HOURS])

    def to_security(entity: int) -> None:
        security.submit(entity, demand.security_time[entity])

    def after_checkin(visit: Visit) -> None:
        to_security(visit.entity)

    def arrive(entity: int) -> None:
        if demand.checks_in[entity]:
            checkin.submit(entity, demand.checkin_time[entity], after_checkin)
        else:
            to_security(entity)

    for entity, t in enumerate(demand.arrival.tolist()):
        sim.schedule_at(t, arrive, entity)
    sim.run()
    stranded = sum(v.depart is None for v in checkin.visits + security.visits)
    if stranded:
        raise RuntimeError(f"{stranded} passengers never cleared: the roster closes on a queue")
    return TerminalRun(sim, checkin, security)


def _by_entity(visits: list[Visit]) -> list[Visit]:
    return sorted(visits, key=lambda v: v.entity)


def _hourly(visits: list[Visit]) -> tuple[np.ndarray, np.ndarray]:
    """Time-average queue length in each hour, and mean wait of those who joined in that hour.

    Hours nobody joined report a mean wait of zero, since nobody waited.
    """
    arrive = np.array([v.arrive for v in visits])
    start = np.array([v.start for v in visits])
    lo = 60.0 * np.arange(HOURS)
    overlap = np.clip(np.minimum(start, lo[:, None] + 60) - np.maximum(arrive, lo[:, None]), 0, None)
    queue = overlap.sum(axis=1) / 60
    hour = np.floor(arrive / 60).astype(int)
    today = (hour >= 0) & (hour < HOURS)
    joined = np.bincount(hour[today], minlength=HOURS)
    waited = np.bincount(hour[today], weights=(start - arrive)[today], minlength=HOURS)
    return queue, np.divide(waited, joined, out=np.zeros(HOURS), where=joined > 0)


def terminal_kpis(config: TerminalConfig, demand: Demand, run: TerminalRun) -> dict[str, float]:
    """Day-level KPIs plus hourly security profiles keyed `_hHH`.

    Utilisation is busy time over open desk or lane time, so closed hours do not dilute it.
    """
    departures = np.array([f.departure for f in config.schedule])
    checkin_wait = np.array([v.wait for v in run.checkin.visits])
    security = _by_entity(run.security.visits)
    security_wait = np.array([v.wait for v in security])
    airside = np.array([v.depart for v in security])
    late = airside > departures[demand.flight] - config.security_cutoff
    kpis = {
        "passengers": float(len(demand)),
        "checkin_wait_mean": float(checkin_wait.mean()),
        "checkin_wait_p95": float(np.percentile(checkin_wait, 95)),
        "security_wait_mean": float(security_wait.mean()),
        "security_wait_p95": float(np.percentile(security_wait, 95)),
        "security_queue_peak": float(run.security.in_queue.peak),
        "time_to_airside_mean": float((airside - demand.arrival).mean()),
        "late_share": float(late.mean()),
        "checkin_utilisation": run.checkin.utilisation(),
        "security_utilisation": run.security.utilisation(),
        "checkin_desk_hours": run.checkin.server_time() / 60,
        "security_lane_hours": run.security.server_time() / 60,
        "littles_residual": max(
            run.checkin.littles_law_residual(), run.security.littles_law_residual()
        ),
    }
    queue, wait = _hourly(security)
    for h in range(HOURS):
        kpis[f"security_queue_h{h:02d}"] = float(queue[h])
        kpis[f"security_wait_h{h:02d}"] = float(wait[h])
    # Worst over open hours only. Anyone arriving before opening waits for it whatever the roster.
    open_hours = np.array(as_roster(config.security_lanes)) > 0
    kpis["security_wait_worst_hour"] = float(wait[open_hours].max())
    return kpis


def terminal_model(config: TerminalConfig, streams: Streams) -> dict[str, float]:
    demand = generate_demand(config, streams)
    return terminal_kpis(config, demand, simulate_terminal(config, demand))
