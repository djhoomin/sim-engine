"""Compare two security staffing levels on the synthetic terminal day.

Both scenarios run with the same base seed, so each replication pair sees identical passengers
and the reported differences are paired. Both experiment records are saved to runs/.

    uv run python scripts/compare_staffing.py --lanes 7 8 --reps 30
"""

import argparse
from dataclasses import replace

from simengine.experiment import paired_difference, run_experiment
from simengine.schedule import synthetic_schedule
from simengine.terminal import TerminalConfig, terminal_model

KPIS = [
    "security_wait_mean",
    "security_wait_p95",
    "security_queue_peak",
    "time_to_airside_mean",
    "checkin_wait_mean",
    "late_share",
    "security_utilisation",
]


def fmt(ci) -> str:
    return f"{ci.mean:9.3f} ± {ci.half_width:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lanes", type=int, nargs=2, default=[7, 8])
    parser.add_argument("--desks", type=int, default=22)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    base = TerminalConfig(
        schedule=synthetic_schedule(), checkin_desks=args.desks, security_lanes=args.lanes[0]
    )
    a = run_experiment(terminal_model, base, args.reps, args.seed)
    b = run_experiment(terminal_model, replace(base, security_lanes=args.lanes[1]), args.reps, args.seed)
    paths = [a.save(), b.save()]

    la, lb = args.lanes
    print(f"{args.reps} replications, seed {args.seed}, 95% intervals, times in minutes\n")
    print(f"{'kpi':22} {f'{la} lanes':>17} {f'{lb} lanes':>17} {'paired difference':>19}")
    for kpi in KPIS:
        diff = paired_difference(a, b, kpi)
        print(f"{kpi:22} {fmt(a.interval(kpi))} {fmt(b.interval(kpi))} {fmt(diff)}")
    print("\nrecords:", *map(str, paths), sep="\n  ")


if __name__ == "__main__":
    main()
