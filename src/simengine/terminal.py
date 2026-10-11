"""A single departures terminal: check-in feeding security.

Demand comes from the flight schedule. Each flight's passenger count is drawn from its seats and
load factor, and each passenger shows up a random lead time before departure. Summing those
show-up curves over the schedule gives the peaky, non-stationary arrival process that a constant
Poisson rate would miss.

All passengers of a replication are generated up front from their own named streams, before any
resource exists. Staffing never touches demand, so two staffing scenarios with the same seed see
the same passengers arriving at the same moments with the same service needs.

The day starts and ends empty. That makes this a terminating simulation with no warm-up to
choose, and it means Little's Law holds exactly on each resource.
"""

from dataclasses import dataclass, field

import numpy as np

from .kernel import Simulation
from .resource import Resource, Visit
from .rng import Streams
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
    checkin_desks: int
    security_lanes: int
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


@dataclass
class TerminalRun:
    sim: Simulation
    checkin: Resource
    security: Resource


def simulate_terminal(config: TerminalConfig, demand: Demand) -> TerminalRun:
    sim = Simulation(start=float(demand.arrival.min()))
    checkin = Resource(sim, "checkin", config.checkin_desks)
    security = Resource(sim, "security", config.security_lanes)

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
    return TerminalRun(sim, checkin, security)


def _by_entity(visits: list[Visit]) -> list[Visit]:
    return sorted(visits, key=lambda v: v.entity)


def terminal_kpis(config: TerminalConfig, demand: Demand, run: TerminalRun) -> dict[str, float]:
    """Day-level KPIs. Utilisation is over the whole day, first arrival to last departure."""
    departures = np.array([f.departure for f in config.schedule])
    checkin_wait = np.array([v.wait for v in run.checkin.visits])
    security = _by_entity(run.security.visits)
    security_wait = np.array([v.wait for v in security])
    airside = np.array([v.depart for v in security])
    late = airside > departures[demand.flight] - config.security_cutoff
    return {
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
        "littles_residual": max(
            run.checkin.littles_law_residual(), run.security.littles_law_residual()
        ),
    }


def terminal_model(config: TerminalConfig, streams: Streams) -> dict[str, float]:
    demand = generate_demand(config, streams)
    return terminal_kpis(config, demand, simulate_terminal(config, demand))
