"""Readers for the per-airport CSVs that scripts/ingest_us.py writes.

The schedule CSV has `date` (YYYY-MM-DD), `departure` (HH:MM) and `seats`. The checkpoint CSV
has `date` (M/D/YYYY, as TSA prints it), `hour` (HH:MM) and `passengers`, one row per checkpoint.
"""

import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def day_range(first: str, last: str) -> list[dt.date]:
    a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def read_checkpoint_hourly(path: str | Path) -> dict[tuple[dt.date, int], int]:
    """Passengers screened per (date, hour), summed over checkpoints. Missing hours had none."""
    hourly: dict[tuple[dt.date, int], int] = defaultdict(int)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = dt.datetime.strptime(r["date"], "%m/%d/%Y").date()
            hourly[(d, int(r["hour"][:2]))] += int(r["passengers"])
    return hourly


@dataclass(frozen=True)
class CalibrationInputs:
    """Several days on one minute clock starting at midnight of the first day."""

    departures: np.ndarray
    seats: np.ndarray
    edges: np.ndarray
    observed: np.ndarray
    hour_of_bin: np.ndarray
    day_of_bin: np.ndarray


def calibration_inputs(
    schedule_csv: str | Path, checkpoint_csv: str | Path, days: list[dt.date], first_open_hour: int
) -> CalibrationInputs:
    """Bin observed counts by hour, folding each day's hours before `first_open_hour` into one bin.

    The checkpoint is closed overnight, so anyone arriving then is screened when it reopens. The
    folded bin is labelled with hour first_open_hour - 1.
    """
    origin = days[0]

    def minute(d: dt.date, hhmm: str) -> float:
        return (d - origin).days * 1440 + 60 * int(hhmm[:2]) + int(hhmm[3:5])

    departures, seats = [], []
    with open(schedule_csv, newline="") as f:
        for r in csv.DictReader(f):
            d = dt.date.fromisoformat(r["date"])
            # The day after the last also counts, since its early flights draw late arrivals.
            if days[0] <= d <= days[-1] + dt.timedelta(days=1):
                departures.append(minute(d, r["departure"]))
                seats.append(int(r["seats"]))

    hourly = read_checkpoint_hourly(checkpoint_csv)
    edges, observed, hour_of_bin, day_of_bin = [0.0], [], [], []
    for i, d in enumerate(days):
        day_of_bin += [i] * (25 - first_open_hour)
        base = (d - origin).days * 1440
        observed.append(sum(hourly[(d, h)] for h in range(first_open_hour)))
        hour_of_bin.append(first_open_hour - 1)
        edges.append(base + 60 * first_open_hour)
        for h in range(first_open_hour, 24):
            observed.append(hourly[(d, h)])
            hour_of_bin.append(h)
            edges.append(base + 60 * (h + 1))
    return CalibrationInputs(
        np.array(departures, float),
        np.array(seats, float),
        np.array(edges, float),
        np.array(observed, float),
        np.array(hour_of_bin),
        np.array(day_of_bin),
    )


def read_wait_bands_hourly(
    path: str | Path, timezone: str, max_gap_minutes: float = 240.0
) -> dict[tuple[dt.date, int], float]:
    """Time-weighted mean of the wait band midpoints per local (date, hour).

    Reads the CSV that scripts/ingest_cph.py writes. Each reading holds until the next one, since
    the source records changes only. A reading followed by a gap longer than `max_gap_minutes` is
    treated as a feed outage and contributes nothing.
    """
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(timezone)
    points = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["t2_from"]:
                when = dt.datetime.fromisoformat(r["observed_utc"]).astimezone(zone)
                points.append((when, (int(r["t2_from"]) + int(r["t2_to"])) / 2))
    total: dict[tuple[dt.date, int], float] = defaultdict(float)
    weight: dict[tuple[dt.date, int], float] = defaultdict(float)
    for (start, wait), (end, _) in zip(points, points[1:]):
        if (end - start).total_seconds() > 60 * max_gap_minutes:
            continue
        t = start
        while t < end:
            hour_end = t.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
            stop = min(end, hour_end)
            minutes = (stop - t).total_seconds() / 60
            total[(t.date(), t.hour)] += wait * minutes
            weight[(t.date(), t.hour)] += minutes
            t = stop
    return {k: total[k] / weight[k] for k in total if weight[k] > 0}
