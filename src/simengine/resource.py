"""Multi-server FIFO resource with per-visit traces and time-weighted statistics."""

from collections import deque
from dataclasses import dataclass
from typing import Callable

from .kernel import Simulation


@dataclass(slots=True)
class Visit:
    """One entity's pass through one resource. Times are None until they happen."""

    entity: int
    service_time: float
    arrive: float
    start: float | None = None
    depart: float | None = None

    @property
    def wait(self) -> float:
        return self.start - self.arrive

    @property
    def sojourn(self) -> float:
        return self.depart - self.arrive


class TimeWeighted:
    """Running integral of a piecewise-constant quantity over time."""

    def __init__(self, now: float, value: float = 0.0):
        self.value = value
        self.area = 0.0
        self._since = now
        self._start = now

    def update(self, now: float, value: float) -> None:
        self.area += self.value * (now - self._since)
        self.value = value
        self._since = now

    def reset(self, now: float) -> None:
        self.update(now, self.value)
        self.area = 0.0
        self._start = now

    def integral(self, now: float) -> float:
        return self.area + self.value * (now - self._since)

    def mean(self, now: float) -> float:
        span = now - self._start
        return self.integral(now) / span if span > 0 else 0.0


class Resource:
    def __init__(self, sim: Simulation, name: str, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.sim = sim
        self.name = name
        self.capacity = capacity
        self.busy = 0
        self.queue: deque[tuple[Visit, Callable[[Visit], None] | None]] = deque()
        self.visits: list[Visit] = []
        self.stats_start = sim.now
        self.in_system = TimeWeighted(sim.now)
        self.in_queue = TimeWeighted(sim.now)
        self.in_service = TimeWeighted(sim.now)

    def submit(
        self,
        entity: int,
        service_time: float,
        on_done: Callable[[Visit], None] | None = None,
    ) -> Visit:
        """Entity arrives now and needs `service_time` of one server. `on_done` fires on departure."""
        visit = Visit(entity=entity, service_time=service_time, arrive=self.sim.now)
        self.visits.append(visit)
        if self.busy < self.capacity:
            self._start(visit, on_done)
        else:
            self.queue.append((visit, on_done))
        self._record()
        return visit

    def _start(self, visit: Visit, on_done: Callable[[Visit], None] | None) -> None:
        self.busy += 1
        visit.start = self.sim.now
        self.sim.schedule(visit.service_time, self._finish, visit, on_done)

    def _finish(self, visit: Visit, on_done: Callable[[Visit], None] | None) -> None:
        visit.depart = self.sim.now
        self.busy -= 1
        if self.queue:
            self._start(*self.queue.popleft())
        self._record()
        if on_done is not None:
            on_done(visit)

    def _record(self) -> None:
        now = self.sim.now
        self.in_system.update(now, self.busy + len(self.queue))
        self.in_queue.update(now, len(self.queue))
        self.in_service.update(now, self.busy)

    def reset_stats(self) -> None:
        """Start a fresh observation window now, for example at the end of warm-up."""
        now = self.sim.now
        self.stats_start = now
        for acc in (self.in_system, self.in_queue, self.in_service):
            acc.reset(now)

    def utilisation(self) -> float:
        return self.in_service.mean(self.sim.now) / self.capacity

    def littles_law_residual(self) -> float:
        """Relative gap between the time-average L and the per-visit sojourns in the window.

        Over the window [stats_start, now], the area under the number in system must equal the
        sum of each visit's time in system clipped to that window. That is Little's Law in its
        exact sample-path form, so a non-zero residual means the bookkeeping is wrong.
        """
        now = self.sim.now
        area = self.in_system.integral(now)
        clipped = 0.0
        for v in self.visits:
            end = v.depart if v.depart is not None else now
            clipped += max(0.0, min(end, now) - max(v.arrive, self.stats_start))
        scale = max(abs(area), 1.0)
        return abs(area - clipped) / scale
