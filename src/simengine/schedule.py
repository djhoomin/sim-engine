"""Flight schedules: the input that drives passenger demand.

Times are minutes after midnight. A real schedule loads from CSV with the same shape as the
synthetic one, so swapping synthetic for real data changes the input and nothing else.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Flight:
    departure: float
    seats: int


def load_schedule(path: str | Path, date: str | None = None) -> tuple[Flight, ...]:
    """Read a CSV with columns `departure` (HH:MM) and `seats`.

    With `date`, keep only rows whose `date` column matches, for files holding many days.
    """
    flights = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if date is not None and row["date"] != date:
                continue
            hours, minutes = row["departure"].split(":")
            flights.append(Flight(departure=60 * int(hours) + int(minutes), seats=int(row["seats"])))
    return tuple(sorted(flights, key=lambda fl: fl.departure))


def synthetic_schedule(flights: int = 60, seed: int = 2026) -> tuple[Flight, ...]:
    """A short-haul European day with morning, midday and evening banks.

    The schedule is fixed by its own seed, not by the replication, because a real timetable does
    not change between replications either.
    """
    rng = np.random.default_rng(seed)
    banks = np.array([7.5, 12.5, 18.0]) * 60
    bank = rng.choice(len(banks), size=flights, p=[0.4, 0.25, 0.35])
    departures = np.clip(rng.normal(banks[bank], 60), 6 * 60, 22.5 * 60)
    departures = np.round(departures / 5) * 5
    seats = rng.choice([150, 180, 186, 220], size=flights, p=[0.3, 0.35, 0.2, 0.15])
    return tuple(
        sorted(
            (Flight(float(d), int(s)) for d, s in zip(departures, seats)),
            key=lambda fl: fl.departure,
        )
    )
