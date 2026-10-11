"""A single multi-server station fed by a renewal arrival process.

This is the validation workhorse: with exponential gaps and services it is M/M/c, and with
other samplers it is G/G/c. The airport terminal model will be built from the same parts.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .kernel import Simulation
from .resource import Resource
from .rng import Streams

Sampler = Callable[[np.random.Generator, int], np.ndarray]


def draws(rng: np.random.Generator, sampler: Sampler, batch: int = 4096) -> Iterator[float]:
    """Yield values one at a time while drawing from numpy in batches.

    The values used are always a prefix of the stream, so batching does not break common
    random numbers.
    """
    while True:
        yield from sampler(rng, batch).tolist()


def exponential(rate: float) -> Sampler:
    return lambda rng, n: rng.exponential(1 / rate, n)


def simulate_station(
    gaps: Iterator[float],
    services: Iterator[float],
    servers: int,
    horizon: float,
    warmup: float = 0.0,
    drain: bool = False,
) -> Resource:
    """Run one station. Arrivals stop at `horizon`. With `drain`, run on until the station is empty.

    Each entity takes its service requirement from `services` at arrival, in arrival order. That
    keeps the n-th customer's service time identical across scenarios with different staffing.
    """
    sim = Simulation()
    station = Resource(sim, "station", servers)
    entity = 0

    def arrive() -> None:
        nonlocal entity
        station.submit(entity, next(services))
        entity += 1
        nxt = sim.now + next(gaps)
        if nxt <= horizon:
            sim.schedule_at(nxt, arrive)

    first = next(gaps)
    if first <= horizon:
        sim.schedule_at(first, arrive)
    if warmup > 0:
        sim.schedule_at(warmup, station.reset_stats)
    sim.run(until=None if drain else horizon)
    return station


@dataclass(frozen=True)
class MMcConfig:
    arrival_rate: float
    service_rate: float
    servers: int
    horizon: float
    warmup: float


def station_kpis(station: Resource, warmup: float) -> dict[str, float]:
    started = [v for v in station.visits if v.arrive >= warmup and v.start is not None]
    done = [v for v in started if v.depart is not None]
    span = station.sim.now - station.stats_start
    return {
        "Wq": float(np.mean([v.wait for v in started])),
        "W": float(np.mean([v.sojourn for v in done])),
        "Lq": station.in_queue.mean(station.sim.now),
        "L": station.in_system.mean(station.sim.now),
        "utilisation": station.utilisation(),
        "arrival_rate": sum(1 for v in station.visits if v.arrive >= warmup) / span,
        "littles_residual": station.littles_law_residual(),
    }


def mmc_model(config: MMcConfig, streams: Streams) -> dict[str, float]:
    gaps = draws(streams.stream("arrivals"), exponential(config.arrival_rate))
    services = draws(streams.stream("service"), exponential(config.service_rate))
    station = simulate_station(gaps, services, config.servers, config.horizon, config.warmup)
    return station_kpis(station, config.warmup)
