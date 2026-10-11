import pytest

from simengine.kernel import Simulation


def test_events_fire_in_time_order_with_fifo_ties():
    sim = Simulation()
    log = []
    sim.schedule(2.0, log.append, "c")
    sim.schedule(1.0, log.append, "a")
    sim.schedule(1.0, log.append, "b")
    sim.run()
    assert log == ["a", "b", "c"]
    assert sim.now == 2.0


def test_run_until_stops_and_advances_clock():
    sim = Simulation()
    log = []
    sim.schedule(1.0, log.append, 1)
    sim.schedule(5.0, log.append, 5)
    sim.run(until=3.0)
    assert log == [1]
    assert sim.now == 3.0
    assert sim.pending == 1


def test_cannot_schedule_into_the_past():
    sim = Simulation()
    with pytest.raises(ValueError):
        sim.schedule(-1.0, print)
