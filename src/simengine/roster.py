"""Hourly staffing rosters.

A roster is 24 server counts, one per hour of the day, where hour h covers minutes 60h to 60h + 60.
A plain integer means the same count around the clock.
"""

import math
from collections.abc import Sequence

import numpy as np

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

    `offered_load[h]` is the expected busy server time in hour h, in server hours. This is the
    classic stationary independent period by period heuristic. It ignores queues carried over
    from the previous hour, which is exactly the kind of thing the simulation is there to expose.
    """
    open_from, open_until = open_hours
    return tuple(
        # The tolerance stops float noise such as 2.1 / 0.7 = 3.0000000000000004 adding a server.
        max(minimum, math.ceil(load / target_utilisation - 1e-9)) if open_from <= h < open_until else 0
        for h, load in enumerate(np.asarray(offered_load, dtype=float))
    )
