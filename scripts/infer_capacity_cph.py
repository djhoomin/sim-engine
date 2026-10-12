"""Infer Kastrup's security capacity through the day from observed waits, then test it.

Demand comes from each day's real departures with an assumed show-up profile and effective load
factor, since Kastrup publishes no checkpoint counts. Capacity is the unknown. Working forward
from the first hour, each hour's lane count is the one whose simulated mean wait best matches the
observed mean wait over the fit days, with earlier hours fixed at what they inferred and later
hours left ample. Common random numbers keep candidates comparable. The inferred roster is then
run on held-out days and scored against their observed waits.

A simulated day runs from 03:00 to 03:00, so flights just after midnight bring their passengers
the evening before. OpenSky times are takeoff, so the show-up profile is relative to takeoff.
The observed "0-2 min" band means no queue, so one minute is taken off every observed midpoint.

    uv run python scripts/infer_capacity_cph.py data/cph/schedule_cph.csv data/cph/waits.csv \\
        --fit 2025-09-01 2025-09-14 --test 2025-09-15 2025-09-30
"""

import argparse
import datetime as dt

import numpy as np

from simengine.calibration import fit_scores
from simengine.rng import Streams
from simengine.sources import day_flights, day_range, read_departures_by_day, read_wait_bands_hourly
from simengine.terminal import ServiceTime, TerminalConfig, generate_demand, simulate_terminal

DAY_START = 3  # hour a simulated day starts
HOURS = range(DAY_START, 24)  # hours whose waits are inferred and scored
AMPLE = 60  # lanes in hours not yet inferred
NO_QUEUE = 1.0  # midpoint of the "0-2 min" band


def hourly_wait(run, hour: int) -> float:
    """Mean wait of passengers who joined security in `hour` and have started by the time the run stopped."""
    waits = [v.wait for v in run.security.visits if 60 * hour <= v.arrive < 60 * (hour + 1) and v.start is not None]
    return float(np.mean(waits)) if waits else 0.0


class Day:
    """One day's demand per replication, drawn once so every roster sees the same passengers."""

    def __init__(self, flights, args, reps: int, seed: int):
        self.config = TerminalConfig(
            schedule=flights,
            checkin_desks=1,
            security_lanes=AMPLE,
            checkin_share=0.0,
            load_factor=args.load_factor,
            showup_mean=args.showup[0],
            showup_sd=args.showup[1],
            security_time=ServiceTime(args.service, 0.5),
        )
        self.demand = [generate_demand(self.config, Streams(seed, r)) for r in range(reps)]

    def runs(self, lanes: list[int], until: float | None = None):
        config = TerminalConfig(**{**self.config.__dict__, "security_lanes": tuple(lanes)})
        return [simulate_terminal(config, d, until=until) for d in self.demand]


def infer(days: list[Day], observed: np.ndarray, max_lanes: int) -> list[int]:
    lanes = [AMPLE] * 24
    for i, hour in enumerate(HOURS):
        target = observed[i]

        def mean_wait(n: int) -> float:
            trial = lanes[:hour] + [n] + [AMPLE] * (23 - hour)
            return float(np.mean([hourly_wait(r, hour) for d in days for r in d.runs(trial, until=60 * (hour + 2))]))

        # Waits fall as lanes rise, so find the fewest lanes whose wait is at or below target.
        lo, hi = 1, max_lanes
        while lo < hi:
            mid = (lo + hi) // 2
            if mean_wait(mid) <= target:
                hi = mid
            else:
                lo = mid + 1
        if lo > 1 and abs(mean_wait(lo - 1) - target) < abs(mean_wait(lo) - target):
            lo -= 1
        lanes[hour] = lo
        print(f"  {hour:02d}:00  observed {target:5.1f} min  ->  {lo:2d} lanes", flush=True)
    for hour in range(DAY_START):
        lanes[hour] = lanes[23]
    return lanes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schedule")
    parser.add_argument("waits")
    parser.add_argument("--fit", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--test", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--load-factor", type=float, default=0.68, help="occupancy times share passing security")
    parser.add_argument("--showup", type=float, nargs=2, default=[105.0, 55.0], metavar=("MEAN", "SD"), help="minutes before takeoff")
    parser.add_argument("--service", type=float, default=0.4, help="minutes per passenger per lane")
    parser.add_argument("--max-lanes", type=int, default=60)
    parser.add_argument("--search-reps", type=int, default=3)
    parser.add_argument("--test-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    schedule = read_departures_by_day(args.schedule)
    waits = read_wait_bands_hourly(args.waits, "Europe/Copenhagen")
    queue_wait = lambda day, h: max(waits[(day, h)] - NO_QUEUE, 0.0)

    fit_days = [d for d in day_range(*args.fit) if schedule[d] and schedule[d + dt.timedelta(days=1)]]
    observed = np.array([np.mean([queue_wait(d, h) for d in fit_days]) for h in HOURS])
    print(f"inferring lanes on {len(fit_days)} days, load factor {args.load_factor}, show-up {args.showup[0]:.0f} ± {args.showup[1]:.0f} min before takeoff, {60 / args.service:.0f} passengers per lane hour")
    sims = [Day(day_flights(schedule, d), args, args.search_reps, args.seed) for d in fit_days]
    lanes = infer(sims, observed, args.max_lanes)

    test_days = [d for d in day_range(*args.test) if schedule[d] and schedule[d + dt.timedelta(days=1)]]
    sim_profile, obs_profile, per_day = [], [], []
    for day in test_days:
        runs = Day(day_flights(schedule, day), args, args.test_reps, args.seed + 1).runs(lanes)
        sim = np.array([np.mean([hourly_wait(r, h) for r in runs]) for h in HOURS])
        obs = np.array([queue_wait(day, h) for h in HOURS])
        sim_profile.append(sim)
        obs_profile.append(obs)
        per_day.append(np.corrcoef(sim, obs)[0, 1])
    sim_mean, obs_mean = np.mean(sim_profile, axis=0), np.mean(obs_profile, axis=0)
    s = fit_scores(obs_mean, sim_mean)
    print(f"\nheld out {len(test_days)} days, {args.test[0]} to {args.test[1]}")
    print(f"  mean daily profile: r2 {s['r2']:.2f}, mae {s['mae']:.2f} min")
    print(f"  single days: hourly correlation median {np.median(per_day):.2f}, range {min(per_day):.2f} to {max(per_day):.2f}")
    demand = np.mean([np.histogram(d.demand[0].arrival, 60.0 * np.arange(25))[0] for d in sims], axis=0)
    print("\n hour  lanes  capacity/h  arrivals/h  utilisation  wait observed  wait simulated (test)")
    for i, h in enumerate(HOURS):
        cap = lanes[h] * 60 / args.service
        print(f"   {h:02d}  {lanes[h]:5d}  {cap:10.0f}  {demand[h]:10.0f}  {demand[h] / cap:11.2f}  {obs_mean[i]:13.1f}  {sim_mean[i]:14.1f}")
    print(f"lane hours {sum(lanes[h] for h in HOURS)}")


if __name__ == "__main__":
    main()
