"""Event-scheduling kernel: a clock and a time-ordered event list."""

import heapq
import itertools
from typing import Any, Callable


class Simulation:
    def __init__(self):
        self.now = 0.0
        self._events: list[tuple[float, int, Callable[..., Any], tuple]] = []
        # Tie-breaker so events at equal times fire in the order they were scheduled.
        self._seq = itertools.count()

    def schedule(self, delay: float, action: Callable[..., Any], *args: Any) -> None:
        if delay < 0:
            raise ValueError(f"cannot schedule into the past (delay={delay})")
        heapq.heappush(self._events, (self.now + delay, next(self._seq), action, args))

    def schedule_at(self, time: float, action: Callable[..., Any], *args: Any) -> None:
        self.schedule(time - self.now, action, *args)

    def run(self, until: float | None = None) -> None:
        """Fire events in time order. With `until`, stop there and leave the clock at `until`."""
        while self._events:
            time, _, action, args = self._events[0]
            if until is not None and time > until:
                break
            heapq.heappop(self._events)
            self.now = time
            action(*args)
        if until is not None and until > self.now:
            self.now = until

    @property
    def pending(self) -> int:
        return len(self._events)
