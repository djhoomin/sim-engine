"""What is known or assumed about Copenhagen Airport's central security, in one place.

Terminals 2 and 3 share one central checkpoint, until a new Terminal 3 building opens with its own
security, expected around 2028. CT lanes replaced conventional X-ray lanes in phases from April
2025, with eleven CT lanes in service from September 2025, until the new 20-lane CT checkpoint
opened on 19 May 2026. During the transition, automated gates sent passengers either to the X-ray
lanes, at most two of four ever staffed, or to a CT queue, one per pair of CT machines.
"""

from .checkpoint import LaneGroup, Layout
from .terminal import ServiceTime, TerminalConfig, TransitLine

TIMEZONE = "Europe/Copenhagen"
DAY_START = 3  # an operating day runs from 03:00 to 03:00
NO_QUEUE = 1.0  # midpoint of the feed's "0-2 min" band, which means no queue

# 2019 arrival split, 33% train and 27% metro, on a stylised timetable.
RAIL = (TransitLine("metro", 0.27, headway=5.0), TransitLine("train", 0.33, headway=15.0, offset=7.0))

# Seat occupancy reported by CPH times the 85% assumed to pass central security, the rest being
# transfers who do not. September from CPH's 80% occupancy, January from 2.226 million passengers
# against the schedule's seats, about 77%.
LOAD_FACTOR = {"2025-09": 0.68, "2026-01": 0.65}


def transition_layout(ct_speed: float) -> Layout:
    """Eleven CT lanes in queues of two, then the X-ray lanes, two of them ever staffed."""
    ct = [LaneGroup(f"ct{i}", 2, ct_speed) for i in range(1, 6)] + [LaneGroup("ct6", 1, ct_speed)]
    return Layout((*ct, LaneGroup("xray", 2)))


def day_config(
    flights,
    load_factor: float,
    showup: tuple[float, float] = (105.0, 55.0),
    service: float = 0.4,
    transit: bool = False,
) -> TerminalConfig:
    """Everyone straight to security, show-up relative to takeoff since OpenSky times are takeoff."""
    return TerminalConfig(
        schedule=flights,
        checkin_desks=1,
        security_lanes=1,
        checkin_share=0.0,
        load_factor=load_factor,
        showup_mean=showup[0],
        showup_sd=showup[1],
        security_time=ServiceTime(service, 0.5),
        transit=RAIL if transit else (),
    )
