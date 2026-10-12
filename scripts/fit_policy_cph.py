"""Fit queue-responsive staffing to Kastrup's observed waits, then test it on held-out days.

Same demand as infer_capacity_cph.py: each day's real departures, an assumed show-up profile and
an effective load factor. Instead of a lane count per hour, security follows a QueuePolicy, which
opens lanes when the estimated wait passes one threshold and closes one below another, within the
lanes available. A grid over the two thresholds and the lanes available is scored on every hour of
every fit day, so days that differ count, not only the average day. Replications share demand
across the grid, so candidates differ only in policy.

Lanes available is fitted rather than assumed because the checkpoint was being rebuilt through
both data periods. CT lanes replaced conventional X-ray lanes in phases from April 2025 until the
new 20-lane CT checkpoint opened on 19 May 2026, so the number in service and their speed changed.

    uv run python scripts/fit_policy_cph.py data/cph/schedule_cph.csv data/cph/waits.csv \\
        --fit 2025-09-01 2025-09-14 --test 2025-09-15 2025-09-29
"""

import argparse
import datetime as dt
import itertools

import numpy as np

from simengine.calibration import fit_scores
from simengine.control import QueuePolicy
from simengine.rng import Streams
from simengine.sources import day_flights, day_range, read_departures_by_day, read_wait_bands_hourly
from simengine.terminal import ServiceTime, TerminalConfig, generate_demand, simulate_terminal, terminal_kpis

HOURS = list(range(3, 24))
NO_QUEUE = 1.0  # midpoint of the "0-2 min" band


def base_config(flights, args) -> TerminalConfig:
    return TerminalConfig(
        schedule=flights,
        checkin_desks=1,
        security_lanes=1,
        checkin_share=0.0,
        load_factor=args.load_factor,
        showup_mean=args.showup[0],
        showup_sd=args.showup[1],
        security_time=ServiceTime(args.service, 0.5),
    )


def simulate(day_demand, base: TerminalConfig, policy: QueuePolicy, lanes: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean hourly wait and lanes open over replications for one day."""
    config = TerminalConfig(**{**base.__dict__, "security_lanes": lanes, "security_policy": policy})
    kpis = [terminal_kpis(config, d, simulate_terminal(config, d)) for d in day_demand]
    wait = np.array([[k[f"security_wait_h{h:02d}"] for h in HOURS] for k in kpis]).mean(axis=0)
    open_ = np.array([[k[f"security_lanes_h{h:02d}"] for h in HOURS] for k in kpis]).mean(axis=0)
    return wait, open_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schedule")
    parser.add_argument("waits")
    parser.add_argument("--fit", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--test", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--load-factor", type=float, default=0.68)
    parser.add_argument("--showup", type=float, nargs=2, default=[105.0, 55.0], metavar=("MEAN", "SD"))
    parser.add_argument("--service", type=float, default=0.4, help="minutes per passenger per lane")
    parser.add_argument("--open-above", type=float, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--close-below", type=float, nargs="+", default=[1, 2, 4])
    parser.add_argument("--lanes", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument("--review", type=float, default=5.0, help="minutes between looks at the queue")
    parser.add_argument("--delay", type=float, default=5.0, help="minutes to open a lane")
    parser.add_argument("--fit-reps", type=int, default=2)
    parser.add_argument("--test-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    schedule = read_departures_by_day(args.schedule)
    waits = read_wait_bands_hourly(args.waits, "Europe/Copenhagen")
    usable = lambda d: bool(schedule[d]) and bool(schedule[d + dt.timedelta(days=1)])
    observed = lambda d: np.array([max(waits[(d, h)] - NO_QUEUE, 0.0) for h in HOURS])

    fit_days = [d for d in day_range(*args.fit) if usable(d)]
    fit = []
    for d in fit_days:
        base = base_config(day_flights(schedule, d), args)
        fit.append((base, [generate_demand(base, Streams(args.seed, r)) for r in range(args.fit_reps)], observed(d)))

    grid = [
        (o, c, n)
        for o, c, n in itertools.product(args.open_above, args.close_below, args.lanes)
        if c < o
    ]
    print(f"fitting {len(grid)} policies on {len(fit_days)} days, {args.fit[0]} to {args.fit[1]}")
    results = []
    for o, c, n in grid:
        policy = QueuePolicy(o, c, args.review, args.delay, max_step=2, min_servers=2)
        sims = [simulate(demand, base, policy, n)[0] for base, demand, _ in fit]
        sse = float(sum(((s - obs) ** 2).sum() for s, (_, _, obs) in zip(sims, fit)))
        results.append((sse, o, c, n))
        print(f"  open above {o:4.0f}  close below {c:3.0f}  lanes {n:3d}  error {sse:9.0f}", flush=True)
    sse, o, c, n = min(results)
    best = QueuePolicy(o, c, args.review, args.delay, max_step=2, min_servers=2)
    print(f"\nbest: open above {o:.0f} min, close below {c:.0f} min, {n} lanes available")

    test_days = [d for d in day_range(*args.test) if usable(d)]
    sims, obs, lanes_open, per_day = [], [], [], []
    for d in test_days:
        base = base_config(day_flights(schedule, d), args)
        demand = [generate_demand(base, Streams(args.seed + 1, r)) for r in range(args.test_reps)]
        wait, open_ = simulate(demand, base, best, n)
        sims.append(wait)
        obs.append(observed(d))
        lanes_open.append(open_)
        per_day.append(np.corrcoef(wait, obs[-1])[0, 1])
    sim_mean, obs_mean, lanes_mean = (np.mean(x, axis=0) for x in (sims, obs, lanes_open))
    s = fit_scores(obs_mean, sim_mean)
    day_mae = float(np.mean(np.abs(np.array(sims) - np.array(obs))))
    print(f"held out {len(test_days)} days, {args.test[0]} to {args.test[1]}")
    print(f"  mean daily profile: r2 {s['r2']:.2f}, mae {s['mae']:.2f} min")
    print(f"  single days: mae {day_mae:.2f} min, hourly correlation median {np.median(per_day):.2f}, range {min(per_day):.2f} to {max(per_day):.2f}")
    print("\n hour  wait observed  wait simulated  lanes open (simulated)")
    for i, h in enumerate(HOURS):
        print(f"   {h:02d}  {obs_mean[i]:13.1f}  {sim_mean[i]:14.1f}  {lanes_mean[i]:10.1f}")
    print(f"lane hours a day, simulated: {lanes_mean.sum():.0f}")


if __name__ == "__main__":
    main()
