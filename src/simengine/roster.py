"""Hourly staffing rosters.

A roster is 24 server counts, one per hour of the day, where hour h covers minutes 60h to 60h + 60.
A plain integer means the same count around the clock.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

HOURS = 24
Roster = tuple[int, ...]


def as_roster(staffing: int | Sequence[int]) -> Roster:
    if isinstance(staffing, int):
        return (staffing,) * HOURS
    roster = tuple(int(n) for n in staffing)
    if len(roster) != HOURS or min(roster) < 0:
        raise ValueError("a roster needs 24 non-negative hourly counts")
    return roster


def shift(lanes: int, open_hour: int, close_hour: int) -> Roster:
    """`lanes` servers from `open_hour` up to `close_hour`, closed otherwise."""
    return tuple(lanes if open_hour <= h < close_hour else 0 for h in range(HOURS))


def server_hours(roster: Roster) -> int:
    return sum(roster)


def staff_to_load(
    offered_load: Sequence[float],
    target_utilisation: float,
    minimum: int = 1,
    open_hours: tuple[int, int] = (0, HOURS),
) -> Roster:
    """Enough servers each hour to keep utilisation at or below the target.

    `offered_load` is the expected number of busy servers, either one value per hour or several
    per hour (96 for quarter hours, say). With several, each hour is sized to its busiest
    interval. That matters when demand climbs or falls steeply inside an hour, since an hourly
    average then hides a stretch where arrivals outrun capacity. Hours are otherwise treated
    independently, so any queue carried across an hour boundary is left for the simulation to
    expose.
    """
    load = np.asarray(offered_load, dtype=float)
    if len(load) % HOURS:
        raise ValueError("offered load needs a whole number of intervals per hour")
    peak = load.reshape(HOURS, -1).max(axis=1)
    open_from, open_until = open_hours
    return tuple(
        # The tolerance stops float noise such as 2.1 / 0.7 = 3.0000000000000004 adding a server.
        max(minimum, math.ceil(peak[h] / target_utilisation - 1e-9)) if open_from <= h < open_until else 0
        for h in range(HOURS)
    )


@dataclass(frozen=True)
class Shift:
    start: int
    hours: int
    lanes: int


def cover_with_shifts(
    requirement: Roster,
    min_hours: int,
    max_hours: int,
    max_step: int | None = None,
) -> tuple[Roster, tuple[Shift, ...]]:
    """The cheapest roster built from whole shifts that meets `requirement` every hour.

    Each shift keeps one server open for `min_hours` to `max_hours` consecutive hours, inside the
    span from the first to the last hour with a non-zero requirement. With `max_step`, the number
    open may change by at most that many from one hour to the next inside the span. Opening at the
    start of the day and closing at the end are exempt.

    This is the classic shift covering integer program. Here it is small enough to solve exactly
    in milliseconds. Ties between equally cheap rosters go to the one with fewer shifts.
    """
    need = np.array(as_roster(requirement))
    open_hours = np.flatnonzero(need)
    if len(open_hours) == 0:
        return (0,) * HOURS, ()
    first, last = int(open_hours[0]), int(open_hours[-1])
    if last - first + 1 < min_hours:
        raise ValueError("the open span is shorter than the minimum shift")

    shifts = [
        (start, length)
        for length in range(min_hours, max_hours + 1)
        for start in range(first, last - length + 2)
    ]
    cover = np.zeros((HOURS, len(shifts)))
    for j, (start, length) in enumerate(shifts):
        cover[start : start + length, j] = 1

    constraints = [LinearConstraint(cover[first : last + 1], lb=need[first : last + 1])]
    if max_step is not None and last > first:
        step = cover[first + 1 : last + 1] - cover[first:last]
        constraints.append(LinearConstraint(step, lb=-max_step, ub=max_step))
    lengths = np.array([length for _, length in shifts], dtype=float)
    result = milp(
        c=lengths + 1e-3,
        constraints=constraints,
        integrality=np.ones(len(shifts)),
        bounds=Bounds(0, np.inf),
    )
    if not result.success:
        raise ValueError(f"no roster meets the constraints: {result.message}")

    counts = np.round(result.x).astype(int)
    roster = tuple(int(n) for n in cover @ counts)
    chosen = tuple(
        Shift(start, length, int(n)) for (start, length), n in zip(shifts, counts) if n > 0
    )
    return roster, chosen
