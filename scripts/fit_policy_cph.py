"""Fit queue-responsive staffing to Kastrup's observed waits, then test it on held-out days.

Same demand as infer_capacity_cph.py: each day's real departures, an assumed show-up profile and
an effective load factor. Instead of a lane count per hour, security follows a QueuePolicy, which
opens lanes when the estimated wait passes one threshold and closes one below another, within the
lanes available. A grid over the two thresholds and the lanes available is scored on every hour of
every fit day, so days that differ count, not only the average day. Replications share demand
across the grid, so candidates differ only in policy.

With --layout kastrup the checkpoint is split the way it ran during the move to CT: automated
gates send each passenger to one queue, either the conventional X-ray lanes, of which at most two
were ever staffed, or one of the CT groups, a queue per pair of CT machines, eleven CT lanes in all
from September 2025. Lanes open CT first with X-ray as overflow, and gates pick the shortest
estimated wait. Lanes available is then the staff limit across the whole checkpoint, and CT lane
speed relative to X-ray joins the grid.

With --transit, 60% of passengers arrive by rail, the 2019 split of 33% train and 27% metro, on a
metro every 5 minutes and a train every 15, each taking the last service that gets them to
security in time. --compare-transit runs the best policy on the test days with and without that
bunching on the same passengers, so the difference is the bunching alone.

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
from simengine import kastrup
from simengine.checkpoint import Layout
from simengine.control import QueuePolicy
from simengine.rng import Streams
from simengine.sources import day_flights, day_range, read_departures_by_day, read_wait_bands_hourly
from simengine.terminal import TerminalConfig, generate_demand, simulate_terminal, terminal_kpis

HOURS = list(range(3, 24))
NO_QUEUE = kastrup.NO_QUEUE
RAIL = kastrup.RAIL


def base_config(flights, args) -> TerminalConfig:
    return kastrup.day_config(flights, args.load_factor, tuple(args.showup), args.service, args.transit)


def simulate(day_demand, base: TerminalConfig, policy: QueuePolicy, lanes: int, layout: Layout | None = None):
    """Mean hourly wait and lanes open over replications for one day."""
    config = TerminalConfig(
        **{**base.__dict__, "security_lanes": lanes, "security_policy": policy, "security_layout": layout}
    )
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
    parser.add_argument("--lanes", type=int, nargs="+", default=[16, 20, 24], help="lanes available, or the staff limit with --layout kastrup")
    parser.add_argument("--layout", choices=["single", "kastrup"], default="single")
    parser.add_argument("--ct-speed", type=float, nargs="+", default=[1.0], help="CT service time relative to X-ray, with --layout kastrup")
    parser.add_argument("--review", type=float, default=5.0, help="minutes between looks at the queue")
    parser.add_argument("--delay", type=float, default=5.0, help="minutes to open a lane")
    parser.add_argument("--transit", action="store_true", help="passengers arriving by rail come in pulses")
    parser.add_argument("--compare-transit", action="store_true", help="paired test-day comparison with and without rail pulses")
    parser.add_argument("--fit-reps", type=int, default=2)
    parser.add_argument("--test-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    schedule = read_departures_by_day(args.schedule)
    waits = read_wait_bands_hourly(args.waits, kastrup.TIMEZONE)
    usable = lambda d: bool(schedule[d]) and bool(schedule[d + dt.timedelta(days=1)])
    observed = lambda d: np.array([max(waits[(d, h)] - NO_QUEUE, 0.0) for h in HOURS])

    fit_days = [d for d in day_range(*args.fit) if usable(d)]
    fit = []
    for d in fit_days:
        base = base_config(day_flights(schedule, d), args)
        fit.append((base, [generate_demand(base, Streams(args.seed, r)) for r in range(args.fit_reps)], observed(d)))

    speeds = args.ct_speed if args.layout == "kastrup" else [None]
    grid = [
        (o, c, n, v)
        for o, c, n, v in itertools.product(args.open_above, args.close_below, args.lanes, speeds)
        if c < o
    ]
    layout_for = lambda v: kastrup.transition_layout(v) if v is not None else None
    print(f"fitting {len(grid)} policies on {len(fit_days)} days, {args.fit[0]} to {args.fit[1]}")
    results = []
    for o, c, n, v in grid:
        policy = QueuePolicy(o, c, args.review, args.delay, max_step=2, min_servers=2)
        sims = [simulate(demand, base, policy, n, layout_for(v))[0] for base, demand, _ in fit]
        sse = float(sum(((s - obs) ** 2).sum() for s, (_, _, obs) in zip(sims, fit)))
        results.append((sse, o, c, n, v))
        speed = f"  CT {60 / (args.service * v):3.0f}/h" if v is not None else ""
        print(f"  open above {o:4.0f}  close below {c:3.0f}  lanes {n:3d}{speed}  error {sse:9.0f}", flush=True)
    sse, o, c, n, v = min(results, key=lambda r: r[0])
    best = QueuePolicy(o, c, args.review, args.delay, max_step=2, min_servers=2)
    speed = f", CT lanes {60 / (args.service * v):.0f} passengers an hour" if v is not None else ""
    print(f"\nbest: open above {o:.0f} min, close below {c:.0f} min, {n} lanes available{speed}")
    for k in sorted({r[3] for r in results}):
        print(f"  best with {k} lanes available: error {min(r[0] for r in results if r[3] == k):.0f}")

    test_days = [d for d in day_range(*args.test) if usable(d)]
    sims, obs, lanes_open, per_day = [], [], [], []
    for d in test_days:
        base = base_config(day_flights(schedule, d), args)
        demand = [generate_demand(base, Streams(args.seed + 1, r)) for r in range(args.test_reps)]
        wait, open_ = simulate(demand, base, best, n, layout_for(v))
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

    if args.compare_transit:
        compare_transit(args, schedule, test_days, best, n, layout_for(v))


def compare_transit(args, schedule, days, policy: QueuePolicy, lanes: int, layout: Layout | None) -> None:
    """Same passengers and policy with and without rail pulses, paired by replication."""
    from simengine.experiment import confidence_interval

    rows = {k: [] for k in ("security_wait_mean", "security_wait_p95", "security_queue_peak", "security_lane_hours")}
    for d in days:
        base = base_config(day_flights(schedule, d), args)
        for r in range(args.test_reps):
            out = []
            for transit in ((), RAIL):
                config = TerminalConfig(**{
                    **base.__dict__, "transit": transit, "security_lanes": lanes,
                    "security_policy": policy, "security_layout": layout,
                })
                demand = generate_demand(config, Streams(args.seed + 2, r))
                out.append(terminal_kpis(config, demand, simulate_terminal(config, demand)))
            for k in rows:
                rows[k].append((out[0][k], out[1][k]))
    print(f"\nrail pulses on {len(days)} test days x {args.test_reps} replications, same passengers, 95% intervals")
    print(f"{'kpi':22} {'no pulses':>10} {'with pulses':>12} {'paired difference':>20}")
    for k, pairs in rows.items():
        a, b = np.array(pairs).T
        ci = confidence_interval(b - a)
        print(f"{k:22} {a.mean():10.2f} {b.mean():12.2f} {ci.mean:12.2f} ± {ci.half_width:.2f}")


if __name__ == "__main__":
    main()
