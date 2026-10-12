"""Fit a staff limit that varies through the day to Kastrup's waits, then test it on held-out days.

The supervisor policy, CT speed and split-queue layout stay at what fit_policy_cph.py found for the
month. What changes is the staff limit, the lanes that can be open, which instead of one value all
day gets one value per block of hours. Blocks are fixed in advance at two hours from 03:00, rather
than placed where held-out days showed misses, so the test stays honest. Working forward in time,
each block's limit is the one whose simulated waits in that block best match the fit days, with
earlier blocks at what they found and later blocks at the flat limit. Runs stop two hours after the
block, which is all its waits can depend on. The result is then compared with the flat limit on
held-out days, the same passengers under both.

    uv run python scripts/fit_staff_cap_cph.py data/cph/schedule_cph.csv data/cph/waits.csv \\
        --fit 2025-09-01 2025-09-14 --test 2025-09-15 2025-09-29 \\
        --load-factor 0.68 --open-above 10 --close-below 2 --ct-speed 0.5
"""

import argparse
import datetime as dt

import numpy as np

from simengine import kastrup
from simengine.calibration import fit_scores
from simengine.control import QueuePolicy
from simengine.rng import Streams
from simengine.sources import day_flights, day_range, read_departures_by_day, read_wait_bands_hourly
from simengine.terminal import TerminalConfig, generate_demand, simulate_terminal

HOURS = list(range(kastrup.DAY_START, 24))


def hourly_waits(run, hours) -> np.ndarray:
    """Mean wait of passengers who joined security in each hour and had started by the end of the run."""
    arrive = np.array([v.arrive for v in run.security.visits])
    start = np.array([v.start if v.start is not None else np.nan for v in run.security.visits])
    out = []
    for h in hours:
        mine = (arrive >= 60 * h) & (arrive < 60 * (h + 1)) & ~np.isnan(start)
        out.append(float((start[mine] - arrive[mine]).mean()) if mine.any() else 0.0)
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schedule")
    parser.add_argument("waits")
    parser.add_argument("--fit", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--test", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--load-factor", type=float, required=True)
    parser.add_argument("--open-above", type=float, required=True)
    parser.add_argument("--close-below", type=float, required=True)
    parser.add_argument("--ct-speed", type=float, required=True, help="CT service time relative to X-ray")
    parser.add_argument("--flat", type=int, default=10, help="the flat staff limit to start from and compare with")
    parser.add_argument("--caps", type=int, nargs=2, default=[3, 12], metavar=("LOW", "HIGH"))
    parser.add_argument("--block", type=int, default=2, help="hours per block")
    parser.add_argument("--fit-reps", type=int, default=2)
    parser.add_argument("--test-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    schedule = read_departures_by_day(args.schedule)
    waits = read_wait_bands_hourly(args.waits, kastrup.TIMEZONE)
    usable = lambda d: bool(schedule[d]) and bool(schedule[d + dt.timedelta(days=1)])
    observed = lambda d, hours: np.array([max(waits[(d, h)] - kastrup.NO_QUEUE, 0.0) for h in hours])
    policy = QueuePolicy(args.open_above, args.close_below, max_step=2, min_servers=2)
    layout = kastrup.transition_layout(args.ct_speed)

    def config(base: TerminalConfig, roster: list[int]) -> TerminalConfig:
        return TerminalConfig(**{
            **base.__dict__, "security_lanes": tuple(roster), "security_policy": policy, "security_layout": layout,
        })

    def prepare(days, seed, reps):
        out = []
        for d in days:
            base = kastrup.day_config(day_flights(schedule, d), args.load_factor)
            out.append((d, base, [generate_demand(base, Streams(seed, r)) for r in range(reps)]))
        return out

    fit = prepare([d for d in day_range(*args.fit) if usable(d)], args.seed, args.fit_reps)
    blocks = [list(range(h, min(h + args.block, 24))) for h in range(kastrup.DAY_START, 24, args.block)]
    roster = [args.flat] * 24
    print(f"fitting a staff limit per {args.block}-hour block on {len(fit)} days, policy open above {args.open_above:.0f} close below {args.close_below:.0f}, CT {60 / (0.4 * args.ct_speed):.0f} an hour")
    for hours in blocks:
        best = None
        for cap in range(args.caps[0], args.caps[1] + 1):
            trial = roster.copy()
            for h in hours:
                trial[h] = cap
            until = 60.0 * (hours[-1] + 3)
            sse = 0.0
            for d, base, demand in fit:
                sim = np.mean([hourly_waits(simulate_terminal(config(base, trial), dem, until=until), hours) for dem in demand], axis=0)
                sse += float(((sim - observed(d, hours)) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, cap)
        for h in hours:
            roster[h] = best[1]
        print(f"  {hours[0]:02d}:00 to {hours[-1] + 1:02d}:00  limit {best[1]:2d}  error {best[0]:7.0f}", flush=True)

    flat = [args.flat] * 24
    test = prepare([d for d in day_range(*args.test) if usable(d)], args.seed + 1, args.test_reps)
    results = {"flat": ([], []), "by shift": ([], [])}
    obs_all = []
    for d, base, demand in test:
        obs = observed(d, HOURS)
        obs_all.append(obs)
        for name, r in (("flat", flat), ("by shift", roster)):
            runs = [simulate_terminal(config(base, r), dem) for dem in demand]
            sim = np.mean([hourly_waits(run, HOURS) for run in runs], axis=0)
            results[name][0].append(sim)
            results[name][1].append(np.corrcoef(sim, obs)[0, 1])
    obs_mean = np.mean(obs_all, axis=0)
    print(f"\nheld out {len(test)} days, {args.test[0]} to {args.test[1]}, same passengers under both limits")
    print(f"{'staff limit':12} {'profile r2':>11} {'profile mae':>12} {'day mae':>8} {'day correlation':>16} {'lane hours limit':>17}")
    for name, (sims, corr) in results.items():
        s = fit_scores(obs_mean, np.mean(sims, axis=0))
        day_mae = float(np.mean(np.abs(np.array(sims) - np.array(obs_all))))
        cap_hours = sum((flat if name == "flat" else roster)[h] for h in HOURS)
        print(f"{name:12} {s['r2']:11.2f} {s['mae']:12.2f} {day_mae:8.2f} {np.median(corr):16.2f} {cap_hours:17d}")
    flat_mean, shift_mean = (np.mean(results[k][0], axis=0) for k in ("flat", "by shift"))
    print("\n hour  limit  wait observed  flat limit  limit by shift")
    for i, h in enumerate(HOURS):
        print(f"   {h:02d}  {roster[h]:5d}  {obs_mean[i]:13.1f}  {flat_mean[i]:10.1f}  {shift_mean[i]:14.1f}")


if __name__ == "__main__":
    main()
