"""Run the terminal model on real days and compare passengers screened per hour with TSA's counts.

Show-up profiles are fitted on one week and the model then runs each day of another week from that
day's real schedule, so every comparison is out of sample. Everyone goes straight to security,
since the fitted lead time is already time of screening before departure.

Lane staffing at the real checkpoint is not public. The roster here sizes each hour to its busiest
ten minutes at 90% utilisation and opens every lane at opening to clear the overnight backlog. The
throughput comparison barely depends on it while waits stay short, but waiting times reported here
are what this roster would give, not what passengers at the airport experienced.

    uv run python scripts/replay_us.py data/us/msy_schedule_2026_07.csv data/us/msy_tsa_2026_07.csv \\
        --fit 2026-07-12 2026-07-18 --days 2026-07-19 2026-07-25 --show 2026-07-23
"""

import argparse
import datetime as dt

import numpy as np

from simengine.calibration import Dispersion, fit_dispersion, fit_scores, fit_showup_grouped
from simengine.experiment import run_experiment
from simengine.roster import HOURS, staff_to_load
from simengine.schedule import load_schedule
from simengine.sources import calibration_inputs, day_range, read_checkpoint_hourly
from simengine.terminal import ServiceTime, ShowUpBand, TerminalConfig, offered_load, terminal_model

OPEN = (3, 23)
FIRST_OPEN_HOUR = 4  # for fitting, hours 00 to 03 fold into one bin


def fit_demand(args: argparse.Namespace) -> tuple[tuple[ShowUpBand, ...], Dispersion]:
    data = calibration_inputs(args.schedule, args.tsa, day_range(*args.fit), FIRST_OPEN_HOUR)
    cuts = np.array(args.bands) * 60
    band = np.searchsorted(cuts, data.departures % 1440, side="right")
    fit = fit_showup_grouped(data.departures, data.seats, band, data.edges, data.observed)
    dispersion = fit_dispersion(
        data.departures, data.seats, band, data.edges, data.observed, data.day_of_bin, fit, data.hour_of_bin
    )
    starts = [0, *args.bands]
    bands = tuple(ShowUpBand(h, g.mean, g.sd, g.load_factor) for h, g in zip(starts, fit.groups))
    return bands, dispersion


def day_config(
    args: argparse.Namespace, date: str, bands: tuple[ShowUpBand, ...], dispersion: Dispersion | None
) -> TerminalConfig:
    config = TerminalConfig(
        schedule=load_schedule(args.schedule, date=date),
        checkin_desks=1,
        security_lanes=1,
        checkin_share=0.0,
        showup_bands=bands,
        security_time=ServiceTime(args.service_mean, 0.5),
        day_sd=dispersion.day_sd if dispersion else 0.0,
        flight_concentration=(
            dispersion.flight_concentration
            if dispersion and np.isfinite(dispersion.flight_concentration)
            else None
        ),
    )
    lanes = list(staff_to_load(offered_load(config, 10)["security"], 0.9, 1, OPEN))
    lanes[OPEN[0]] = max(lanes)
    return TerminalConfig(**{**config.__dict__, "security_lanes": tuple(lanes)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schedule")
    parser.add_argument("tsa")
    parser.add_argument("--fit", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--days", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--bands", type=int, nargs="*", default=[12], help="hours where a new show-up band starts")
    parser.add_argument("--show", help="print the hourly comparison for this day")
    parser.add_argument("--service-mean", type=float, default=0.4, help="minutes per passenger per lane")
    parser.add_argument("--no-dispersion", action="store_true", help="independent passengers only")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    bands, dispersion = fit_demand(args)
    print("demand fitted on", " to ".join(args.fit))
    for b in bands:
        print(f"  from {b.from_hour:02d}:00  {b.mean:.0f} ± {b.sd:.0f} min, load factor {b.load_factor:.3f}")
    k = dispersion.flight_concentration
    print(f"  day multiplier log sd {dispersion.day_sd:.3f}")
    if np.isnan(k):
        print("  flight load variation: not identifiable, the hourly residuals exceed what any load variation gives")
    else:
        print(f"  flight concentration {k:.1f}, flight load factor sd about {np.sqrt(0.8 * 0.2 / (k + 1)):.3f}")
    print(f"  {dispersion.systematic_share:.0%} of squared hourly residuals repeat at the same hour every day")
    if args.no_dispersion:
        dispersion = None

    observed_by_hour = read_checkpoint_hourly(args.tsa)
    hours = range(HOURS)
    print(f"\n{args.reps} replications per day, 95% intervals. Waits are under a hypothetical roster.")
    print(f"{'day':14} {'observed':>9} {'simulated 95% band':>20} {'hourly r2':>10} {'hours in band':>14} {'wait p95':>14}")
    shown = None
    hits = []
    for day in day_range(*args.days):
        config = day_config(args, day.isoformat(), bands, dispersion)
        record = run_experiment(terminal_model, config, args.reps, args.seed)
        record.save()
        sims = np.array([record.kpi(f"security_screened_h{h:02d}") for h in hours])
        observed = np.array([observed_by_hour[(day, h)] for h in hours], float)
        low, high = np.percentile(sims, [2.5, 97.5], axis=1)
        active = observed + sims.mean(axis=1) > 0
        inside = np.mean((observed >= low) & (observed <= high) & active) / np.mean(active)
        totals = sims.sum(axis=0)
        t_low, t_high = np.percentile(totals, [2.5, 97.5])
        mark = "" if t_low <= observed.sum() <= t_high else " *"
        p95 = record.interval("security_wait_p95")
        r2 = fit_scores(observed, sims.mean(axis=1))["r2"]
        hits.append(inside)
        print(
            f"{day.isoformat()} {day.strftime('%a')} {observed.sum():9.0f} {t_low:9.0f} to {t_high:6.0f}{mark:2}"
            f" {r2:8.3f} {inside:14.0%} {p95.mean:7.2f} ± {p95.half_width:4.2f}"
        )
        if args.show == day.isoformat():
            shown = (day, config, observed, sims, low, high, record)

    print(f"mean share of hours inside the 95% band: {np.mean(hits):.0%}. * marks a daily total outside its band")

    if shown:
        day, config, observed, sims, low, high, record = shown
        print(f"\n{day.isoformat()} {day.strftime('%A')}, passengers screened per hour")
        print(" hour  lanes  observed  simulated  95% band     mean wait")
        for h in hours:
            if observed[h] or sims[h].mean() >= 0.5:
                wait = record.kpi(f"security_wait_h{h:02d}").mean()
                print(
                    f"   {h:02d}  {config.security_lanes[h]:5d}  {observed[h]:8.0f}  {sims[h].mean():9.0f}"
                    f"  {low[h]:5.0f} to {high[h]:5.0f}  {wait:8.2f}"
                )


if __name__ == "__main__":
    main()
