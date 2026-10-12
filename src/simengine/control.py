"""Staffing that responds to the queue instead of following a fixed roster.

A supervisor looks at the queue every few minutes and estimates the wait from it as queue length
times mean service time over lanes open. Above one threshold they open lanes, which take a few
minutes to start taking passengers. Below another they close one. Lanes available, the staff on
duty, still cap what can open. Two thresholds instead of a lane count for every hour make this
far better conditioned to fit from observed waits, and it is a policy an operation could follow.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from .kernel import Simulation
from .resource import Resource


@dataclass(frozen=True)
class QueuePolicy:
    open_above: float  # estimated wait in minutes that opens lanes
    close_below: float  # estimated wait in minutes that closes one
    review_every: float = 5.0  # minutes between looks at the queue
    open_delay: float = 5.0  # minutes from deciding to open until the lane takes passengers
    max_step: int = 1  # most lanes opened at one review
    min_servers: int = 1

    def __post_init__(self):
        if self.close_below >= self.open_above:
            raise ValueError("close_below must be below open_above, or lanes would flap")


class QueueController:
    def __init__(
        self,
        sim: Simulation,
        resource: Resource,
        policy: QueuePolicy,
        service_mean: float,
        available: Callable[[float], int],
        stop_after: float,
    ):
        self.sim, self.resource, self.policy = sim, resource, policy
        self.service_mean = service_mean
        self.available = available
        # Keep reviewing until the last arrival has cleared and lanes are back to the minimum, with
        # a day's margin as a backstop.
        self.stop_after, self.give_up = stop_after, stop_after + 24 * 60
        self.pending = 0
        sim.schedule(0.0, self.review)

    def estimated_wait(self) -> float:
        return len(self.resource.queue) * self.service_mean / max(self.resource.capacity, 1)

    def review(self) -> None:
        p, r, now = self.policy, self.resource, self.sim.now
        limit = self.available(now)
        if r.capacity > limit:
            r.set_capacity(max(limit, 0))
        wait = self.estimated_wait()
        if wait > p.open_above:
            needed = math.ceil(len(r.queue) * self.service_mean / p.open_above) - r.capacity - self.pending
            step = min(max(needed, 1), p.max_step, limit - r.capacity - self.pending)
            for _ in range(max(step, 0)):
                self.pending += 1
                self.sim.schedule(p.open_delay, self._open)
        elif wait < p.close_below and self.pending == 0 and r.capacity > p.min_servers:
            r.set_capacity(r.capacity - 1)
        busy = r.queue or r.busy or self.pending or r.capacity > p.min_servers
        if (now < self.stop_after or busy) and now < self.give_up:
            self.sim.schedule(p.review_every, self.review)

    def _open(self) -> None:
        self.pending -= 1
        if self.resource.capacity < self.available(self.sim.now):
            self.resource.set_capacity(self.resource.capacity + 1)
