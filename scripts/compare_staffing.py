"""Compare two security staffing rosters on the synthetic terminal day.

Both scenarios run with the same base seed, so each replication pair sees identical passengers
and the reported differences are paired. Both experiment records are saved to runs/.

Scenarios:
  flatN        N lanes from 03:00 to 21:00, for example flat7
  loadU        lanes per hour sized to the expected load at utilisation U, for example load0.75

    uv run python scripts/compare_staffing.py flat7 load0.75 --reps 30
"""

import argparse
from dataclasses import replace

from simengine.experiment import paired_difference, run_experiment
from simengine.roster import Roster, server_hours, shift, staff_to_load
from simengine.schedule import synthetic_schedule
from simengine.terminal import TerminalConfig, offered_load, terminal_model

OPEN = (3, 21)
KPIS = [
    "security_lane_hours",
    "security_wait_mean",
    "security_wait_p95",
    "security_queue_peak",
    "time_to_airside_mean",
    "late_share",
    "security_utilisation",
]


def roster_for(name: str, config: TerminalConfig) -> Roster:
    if name.startswith("flat"):
        return shift(int(name[4:]), *OPEN)
    if name.startswith("load"):
        return staff_to_load(offered_load(config)["security"], float(name[4:]), 1, OPEN)
    raise SystemExit(f"unknown scenario {name!r}, expected flatN or loadU")


def fmt(ci) -> str:
    return f"{ci.mean:9.3f} ± {ci.half_width:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs=2, metavar="SCENARIO", default=["flat7", "load0.75"])
    parser.add_argument("--desks", type=int, default=22)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    base = TerminalConfig(
        schedule=synthetic_schedule(),
        checkin_desks=shift(args.desks, *OPEN),
        security_lanes=shift(1, *OPEN),
    )
    configs = [replace(base, security_lanes=roster_for(n, base)) for n in args.scenarios]
    a, b = (run_experiment(terminal_model, c, args.reps, args.seed) for c in configs)
    paths = [a.save(), b.save()]

    na, nb = args.scenarios
    print(f"{args.reps} replications, seed {args.seed}, 95% intervals, times in minutes\n")
    print(f"{'kpi':22} {na:>17} {nb:>17} {'paired difference':>19}")
    for kpi in KPIS:
        diff = paired_difference(a, b, kpi)
        print(f"{kpi:22} {fmt(a.interval(kpi))} {fmt(b.interval(kpi))} {fmt(diff)}")

    print(f"\n{'hour':>5} {'lanes':>11} {'mean security wait':>27}")
    for h in range(*OPEN):
        lanes = f"{configs[0].security_lanes[h]:>4} {configs[1].security_lanes[h]:>4}"
        wa, wb = (r.interval(f"security_wait_h{h:02d}").mean for r in (a, b))
        print(f"{h:02d}:00 {lanes:>11} {wa:13.2f} {wb:13.2f}")
    print(f"lane hours {server_hours(configs[0].security_lanes)} vs {server_hours(configs[1].security_lanes)}")
    print("\nrecords:", *map(str, paths), sep="\n  ")


if __name__ == "__main__":
    main()
