"""A security checkpoint made of separate queues, each feeding its own group of lanes.

Automated gates send each passenger to one group. At Kastrup during the move to CT, one group was
the conventional X-ray lanes, and each pair of CT lanes had its own queue. A group can be slower or
faster than the passenger's nominal security time, through its `speed` factor.

Staffing is decided for the checkpoint as a whole. `set_capacity(n)` opens n lanes in the layout's
opening order and closes them in reverse. A group closing to zero stops taking passengers at the
gates but keeps one lane open until its queue is empty, as a lane does when it closes in practice.
The checkpoint offers the same interface as a single Resource, so the queue controller and the
terminal KPIs work on either.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .kernel import Simulation
from .resource import Resource, TimeWeighted, Visit


@dataclass(frozen=True)
class LaneGroup:
    name: str
    lanes: int  # machines sharing this group's queue
    speed: float = 1.0  # service time multiplier, so 0.75 means a quarter faster


@dataclass(frozen=True)
class Layout:
    groups: tuple[LaneGroup, ...]  # in the order lanes open
    # "shortest_wait" sends each passenger to the open queue with the shortest estimated wait.
    # "first_open" sends them to the first group in opening order with a free lane, and only
    # when every lane is busy to the shortest estimated wait.
    routing: str = "shortest_wait"

    def __post_init__(self):
        if self.routing not in ("shortest_wait", "first_open"):
            raise ValueError(f"unknown routing {self.routing!r}")

    @property
    def lanes(self) -> int:
        return sum(g.lanes for g in self.groups)


class Checkpoint:
    def __init__(
        self, sim: Simulation, layout: Layout, capacity: int, service_mean: float, name: str = "security"
    ):
        self.sim, self.layout, self.service_mean, self.name = sim, layout, service_mean, name
        self.groups = [Resource(sim, g.name, 0) for g in layout.groups]
        self.target = [0] * len(self.groups)
        self.visits: list[Visit] = []
        self.group_of: dict[int, str] = {}
        self.in_queue = TimeWeighted(sim.now)
        self.in_system = TimeWeighted(sim.now)
        self.in_service = TimeWeighted(sim.now)
        self.servers = TimeWeighted(sim.now)
        self.stats_start = sim.now
        self.capacity = 0
        self.capacity_log: list[tuple[float, int]] = [(sim.now, 0)]
        for r in self.groups:
            r.on_change.append(self._changed)
        self.set_capacity(capacity)

    # The same view of the checkpoint a single Resource gives.
    @property
    def queue(self) -> list:
        return [item for r in self.groups for item in r.queue]

    @property
    def busy(self) -> int:
        return sum(r.busy for r in self.groups)

    def set_capacity(self, capacity: int) -> None:
        capacity = max(0, min(capacity, self.layout.lanes))
        remaining = capacity
        for i, g in enumerate(self.layout.groups):
            self.target[i] = min(g.lanes, remaining)
            remaining -= self.target[i]
        self.capacity = capacity
        self.capacity_log.append((self.sim.now, capacity))
        for i, r in enumerate(self.groups):
            self._apply(i, r)

    def _apply(self, i: int, r: Resource) -> None:
        # A group told to close keeps one lane until its queue has gone.
        wanted = self.target[i] if self.target[i] > 0 or not r.queue else 1
        if r.capacity != wanted:
            r.set_capacity(wanted)

    def _changed(self, r: Resource) -> None:
        i = self.groups.index(r)
        if self.target[i] == 0 and r.capacity > 0 and not r.queue:
            r.set_capacity(0)
            return
        now = self.sim.now
        self.in_queue.update(now, sum(len(g.queue) for g in self.groups))
        self.in_system.update(now, sum(len(g.queue) + g.busy for g in self.groups))
        self.in_service.update(now, sum(g.busy for g in self.groups))
        self.servers.update(now, sum(max(g.capacity, g.busy) for g in self.groups))

    def _route(self) -> int:
        open_groups = [i for i, t in enumerate(self.target) if t > 0]
        if not open_groups:
            # Nothing staffed: wait in the first group, which opens first.
            return 0
        if self.layout.routing == "first_open":
            for i in open_groups:
                if self.groups[i].busy < self.groups[i].capacity and not self.groups[i].queue:
                    return i
        speed = [g.speed for g in self.layout.groups]
        return min(
            open_groups,
            key=lambda i: (len(self.groups[i].queue) + 1) * speed[i] / max(self.groups[i].capacity, 1),
        )

    def submit(self, entity: int, service_time: float, on_done: Callable[[Visit], None] | None = None) -> Visit:
        i = self._route()
        visit = self.groups[i].submit(entity, service_time * self.layout.groups[i].speed, on_done)
        self.visits.append(visit)
        self.group_of[entity] = self.layout.groups[i].name
        return visit

    def utilisation(self) -> float:
        open_time = self.servers.integral(self.sim.now)
        return self.in_service.integral(self.sim.now) / open_time if open_time > 0 else 0.0

    def server_time(self) -> float:
        return self.servers.integral(self.sim.now)

    def littles_law_residual(self) -> float:
        return max(r.littles_law_residual() for r in self.groups)
